import os
import time
import asyncio
import secrets
import logging
import tempfile
import hashlib
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Tuple

import edge_tts
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from openai import AsyncOpenAI
from livekit import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-core")

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://chat-core-ozzy0gzk.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

OPENROUTER_API_KEYS = [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip()]
OPENROUTER_BASE = "https://openrouter.ai"
OPENROUTER_CHAT_MODELS = ["openai/gpt-oss-120b", "qwen/qwen-2.5-72b-instruct"]

GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_BASE = "https://groq.com"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_CHAT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


class RoomManager:
    def __init__(self):
        self.rooms = {}
        self.lock = asyncio.Lock()

    def _new_code(self):
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(secrets.choice(alphabet) for _ in range(6))

    async def create_room(self):
        async with self.lock:
            code = self._new_code()
            while code in self.rooms:
                code = self._new_code()
            self.rooms[code] = {"created_at": time.time(), "users": {}, "history": []}
            return code

    async def join(self, code, ws, name, lang):
        async with self.lock:
            if code not in self.rooms:
                self.rooms[code] = {"created_at": time.time(), "users": {}, "history": []}
            if len(self.rooms[code]["users"]) >= 50:
                return False
            self.rooms[code]["users"][ws] = {"name": name, "lang": lang}
            return True

    async def update_user_lang(self, code, ws, new_lang):
        async with self.lock:
            if code in self.rooms and ws in self.rooms[code]["users"]:
                self.rooms[code]["users"][ws]["lang"] = new_lang

    async def leave(self, code, ws):
        async with self.lock:
            if code in self.rooms and ws in self.rooms[code]["users"]:
                del self.rooms[code]["users"][ws]
                if not self.rooms[code]["users"]:
                    del self.rooms[code]

    async def get_ws_users(self, code):
        async with self.lock:
            if code not in self.rooms:
                return {}
            return dict(self.rooms[code]["users"])

    async def get_users_list(self, code):
        async with self.lock:
            if code not in self.rooms:
                return []
            return [{"name": u["name"], "lang": u["lang"]} for u in self.rooms[code]["users"].values()]

    async def get_history(self, code):
        async with self.lock:
            if code not in self.rooms:
                return []
            return list(self.rooms[code]["history"])

    async def add_history(self, code, entry):
        async with self.lock:
            if code in self.rooms:
                self.rooms[code]["history"].append(entry)
                self.rooms[code]["history"] = self.rooms[code]["history"][-20:]

    async def broadcast(self, code, message, exclude=None):
        users = await self.get_ws_users(code)
        for ws in users:
            if ws == exclude:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                pass


room_manager = RoomManager()


@asynccontextmanager
async def lifespan(app):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        logger.error("LiveKit keys missing!")
    if not OPENROUTER_API_KEYS:
        logger.warning("OPENROUTER_API_KEYS empty! Will use Groq only.")
    else:
        logger.info(f"{len(OPENROUTER_API_KEYS)} OpenRouter key(s) loaded.")
    if not GROQ_KEYS:
        logger.error("GROQ_API_KEYS empty! Whisper will fail.")
    else:
        logger.info(f"{len(GROQ_KEYS)} Groq key(s) loaded.")
    yield


app = FastAPI(title="Chat-Core API", version="22.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: Dict[str, dict] = {}
SESSION_MAX_MSGS = 16
_api_key_idx = {"i": 0}


def append_session(sid, user_text, reply):
    if not sid:
        return
    entry = _sessions.get(sid, {"msgs": []})
    entry["msgs"].append({"role": "user", "content": (user_text or "")[:1000]})
    entry["msgs"].append({"role": "assistant", "content": (reply or "")[:2000]})
    entry["msgs"] = entry["msgs"][-SESSION_MAX_MSGS:]
    _sessions[sid] = entry


def get_session(sid, last_n=6):
    if not sid:
        return []
    entry = _sessions.get(sid)
    if not entry:
        return []
    return entry["msgs"][-last_n:]


class ChatRequest(BaseModel):
    messages: List[dict]
    is_vision: bool = False
    is_translation: bool = False
    session_id: Optional[str] = None


class MeetingTokenRequest(BaseModel):
    room_name: str
    participant_name: str


class TTSRequest(BaseModel):
    text: str
    lang: str = "ar-TN"
    voice: str = "woman"
    session_id: Optional[str] = None


def get_next_key(keys_list):
    if not keys_list:
        raise HTTPException(503, detail="No API keys available.")
    k = keys_list[_api_key_idx["i"] % len(keys_list)]
    _api_key_idx["i"] += 1
    return k


async def call_llm(messages, temperature=0.7, max_tokens=2000, timeout=30.0, prefer_fast=False):
    errors = []

    async def _try_groq():
        if not GROQ_KEYS:
            return None
        for model in GROQ_CHAT_MODELS:
            key = get_next_key(GROQ_KEYS)
            client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=timeout)
            try:
                r = await client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                return r.choices[0].message.content
            except Exception as e:
                errors.append(f"Groq[{model}]: {e}")
                if "429" in str(e) or "rate_limit" in str(e):
                    await asyncio.sleep(0.5)
                continue
        return None

    async def _try_openrouter():
        if not OPENROUTER_API_KEYS:
            return None
        for model in OPENROUTER_CHAT_MODELS:
            key = get_next_key(OPENROUTER_API_KEYS)
            client = AsyncOpenAI(
                base_url=OPENROUTER_BASE,
                api_key=key,
                timeout=timeout,
                default_headers={"HTTP-Referer": "https://onrender.com", "X-Title": "Chat-Core"}
            )
            try:
                r = await client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                return r.choices[0].message.content
            except Exception as e:
                errors.append(f"OpenRouter[{model}]: {e}")
                if "429" in str(e) or "rate_limit" in str(e):
                    await asyncio.sleep(0.5)
                continue
        return None

    if prefer_fast:
        res = await _try_groq()
        if res:
            return res
        res = await _try_openrouter()
        if res:
            return res
    else:
        res = await _try_openrouter()
        if res:
            return res
        res = await _try_groq()
        if res:
            return res

    raise HTTPException(503, detail="LLM Error: " + " | ".join(errors))


async def smart_chat(messages, is_vision, temperature, max_tokens):
    return await call_llm(messages, temperature, max_tokens, prefer_fast=False)


LANG_NAMES = {
    "ar": "Modern Standard Arabic",
    "ar-TN": "Tunisian Arabic dialect (Derja)",
    "ar-MA": "Moroccan Arabic dialect (Darija)",
    "ar-DZ": "Algerian Arabic dialect",
    "ar-EG": "Egyptian Arabic dialect",
    "ar-SA": "Saudi / Gulf Arabic dialect",
    "ar-LB": "Lebanese / Levantine Arabic dialect",
    "ar-SY": "Syrian Arabic dialect",
    "ar-IQ": "Iraqi Arabic dialect",
    "ar-JO": "Jordanian Arabic dialect",
    "ar-KW": "Kuwaiti Arabic dialect",
    "ar-AE": "Emirati Arabic dialect",
    "fr": "French",
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "tr": "Turkish",
    "ru": "Russian",
    "zh": "Chinese (Mandarin)",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "nl": "Dutch",
    "no": "Norwegian",
    "sv": "Swedish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "ur": "Urdu",
    "id": "Indonesian",
    "ms": "Malay",
    "el": "Greek",
    "ro": "Romanian",
    "cs": "Czech",
    "hu": "Hungarian",
    "th": "Thai",
    "vi": "Vietnamese",
    "fa": "Persian"
}

_trans_cache: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = 3600


def _cache_key(text: str, src: str, tgt: str) -> str:
    return f"{src}:{tgt}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"


async def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not text.strip() or src_lang == tgt_lang:
        return text
    k = _cache_key(text, src_lang, tgt_lang)
    item = _trans_cache.get(k)
    if item and time.time() - item[1] < CACHE_TTL:
        return item[0]

    src_name = LANG_NAMES.get(src_lang, src_lang)
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

