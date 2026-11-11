import os
import re
import time
import asyncio
import secrets
import logging
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Tuple

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from livekit import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-core")

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://chat-core-ozzy0gzk.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

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
        logger.error("⚠️ LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing!")
    else:
        logger.info("✅ LiveKit credentials found.")
    yield

app = FastAPI(title="Chat-Core API", version="21.0 Ultra", lifespan=lifespan)

ALLOWED_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_BASE = "https://api.groq.com/openai/v1"

# النماذج المستقرة حالياً في Groq (مرتبة حسب القوة والسرعة)
GROQ_TEXT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

_sessions: Dict[str, dict] = {}
SESSION_MAX_MSGS = 16

def append_session(sid, user_text, reply):
    if not sid: return
    entry = _sessions.get(sid, {"msgs": []})
    entry["msgs"].append({"role": "user", "content": (user_text or "")[:1000]})
    entry["msgs"].append({"role": "assistant", "content": (reply or "")[:2000]})
    entry["msgs"] = entry["msgs"][-SESSION_MAX_MSGS:]
    _sessions[sid] = entry

def get_session(sid, last_n=6):
    if not sid: return []
    entry = _sessions.get(sid)
    if not entry: return []
    return entry["msgs"][-last_n:]

class ChatRequest(BaseModel):
    messages: List[dict]
    is_vision: bool = False
    is_translation: bool = False
    session_id: Optional[str] = None

class MeetingTokenRequest(BaseModel):
    room_name: str
    participant_name: str

_groq_idx = {"i": 0}
def next_groq_key():
    if not GROQ_KEYS: raise HTTPException(503, detail="لا يوجد مفتاح Groq في إعدادات Render.")
    k = GROQ_KEYS[_groq_idx["i"] % len(GROQ_KEYS)]
    _groq_idx["i"] += 1
    return k

async def call_groq(messages, models, temperature=0.7, max_tokens=2000):
    last_err = "no model tried"
    if not GROQ_KEYS:
        raise Exception("No GROQ_KEYS configured in environment variables.")

    for model in models:
        key = next_groq_key()
        client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=30.0)
        try:
            r = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            logger.info(f"✅ Groq success with model: {model}")
            return r.choices[0].message.content
        except Exception as e:
            last_err = str(e)
            logger.warning(f"Groq model {model} failed: {last_err[:160]}")
            if "rate_limit" in last_err or "429" in last_err:
                await asyncio.sleep(1.0)
            continue
    raise Exception(f"All Groq models failed. Last error: {last_err}")

async def smart_chat(messages, is_vision, temperature, max_tokens):
    if not GROQ_KEYS:
        raise HTTPException(502, detail="مفتاح Groq غير موجود في إعدادات Render. يرجى إضافته في Environment.")
    try:
        return await call_groq(messages, GROQ_TEXT_MODELS, temperature, max_tokens)
    except Exception as e:
        logger.error(f"Smart_chat failed: {e}")
        raise HTTPException(502, detail="فشل الاتصال بالذكاء الاصطناعي. تأكد من أن مفتاح Groq صحيح.")

LANG_NAMES = {"ar": "Arabic", "fr": "French", "en": "English", "de": "German", "tr": "Turkish", "it": "Italian"}

# فلتر الهلوسة الصارم (يمنع الكلمات الغريبة التي ظهرت سابقاً)
HALLUCINATION_BLACKLIST = ["كلمات شائعة", "شائعة تشير", "تخيلها", "ترمزية", "شائعة بالعربية", "الموضوعات في الوصف"]

async def _run_whisper_verbose(temp_path: str, language: str) -> dict:
    key = next_groq_key()
    client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=120.0)
    lang_code = language if language in LANG_NAMES else "ar"
    
    with open(temp_path, "rb") as af:
        resp = await client.audio.transcriptions.create(
            model="whisper-large-v3", 
            file=af, 
            language=lang_code, 
            temperature=0.0, 
            response_format="verbose_json"
        )
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

def _should_reject(result: dict, duration_ms: int) -> Tuple[bool, str]:
    text = (result.get("text") or "").strip()
    if not text or len(text) < 2: return True, "empty"
    
    for bad_word in HALLUCINATION_BLACKLIST:
        if bad_word in text:
            logger.warning(f"Hallucination blocked: {text}")
            return True, "hallucination"

    no_speech = float(result.get("no_speech_prob") or 0.0)
    if no_speech > 0.6: return True, "no_speech"
    
    avg_logprob = float(result.get("avg_logprob") or 0.0)
    if avg_logprob < -1.0: return True, "low_confidence"
    
    return False, "ok"

@app.get("/health")
async def health():
    return {"ok": True, "provider": "groq", "keys_configured": len(GROQ_KEYS), "active_rooms": len(room_manager.rooms)}

@app.post("/rooms/create")
async def create_room_endpoint():
    return {"code": await room_manager.create_room()}

@app.post("/get-token")
async def get_token_endpoint(req: MeetingTokenRequest):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        raise HTTPException(500, detail="LiveKit keys not configured")
    unique_identity = f"{req.participant_name}#{secrets.token_hex(3)}"
    try:
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET).with_identity(unique_identity).with_name(req.participant_name).with_grants(api.VideoGrants(room_join=True, room=req.room_name, can_publish=True, can_subscribe=True, can_publish_data=True))
        return {"token": token.to_jwt(), "url": LIVEKIT_URL}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages: raise HTTPException(400, detail="messages فارغة")
    messages = req.messages
    if req.session_id and not req.is_translation:
        ctx = get_session(req.session_id)
        if ctx and messages and messages[0].get("role") == "system":
            messages = [messages[0]] + ctx + messages[1:]

    temperature = 0.0 if req.is_translation else 0.7
    max_tokens = 800 if req.is_translation else 2000

    try:
        reply = await smart_chat(messages, req.is_vision, temperature, max_tokens)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(502, detail=str(e))

    if req.session_id and not req.is_translation:
        last_user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
        append_session(req.session_id, last_user if isinstance(last_user, str) else "[media]", reply)

    return {"reply": reply}

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), target_lang: str = Form("ar"), language: str = Form(""), duration_ms: int = Form(0)):
    if not GROQ_KEYS: raise HTTPException(503, detail="مفتاح Groq مطلوب لاستخراج النص")
    contents = await file.read()
    if len(contents) < 1000: return {"original": "", "translated": "", "reason": "too_small"}
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name; tmp.write(contents)
    try:
        lang_code = (language or target_lang or "ar").lower()
        result = await _run_whisper_verbose(temp_path, lang_code)
        reject, reason = _should_reject(result, duration_ms)
        if reject:
            return {"original": "", "translated": "", "reason": reason}
        return {"original": result.get("text", "").strip(), "translated": "", "reason": "ok"}
    except Exception as e:
        raise HTTPException(500, detail=f"فشل استخراج النص: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

@app.websocket("/ws/{room_code}")
async def ws_endpoint(websocket: WebSocket, room_code: str):
    room_code = room_code.upper().strip()
    await websocket.accept()
    joined, my_name, my_lang = False, "User", "ar"
    try:
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        except asyncio.TimeoutError: await websocket.close(code=4001); return
        if first.get("type") != "join": await websocket.close(); return
        
        my_name = (first.get("name") or "User")[:30]
        my_lang = first.get("lang", "ar")
        if not await room_manager.join(room_code, websocket, my_name, my_lang):
            await websocket.send_json({"type": "error", "message": "Room full"})
            await websocket.close(); return
        joined = True
        
        users = await room_manager.get_users_list(room_code)
        history_raw = await room_manager.get_history(room_code)
        await websocket.send_json({"type": "joined", "users": users, "history": history_raw})
        await room_manager.broadcast(room_code, {"type": "user_joined", "name": my_name}, exclude=websocket)
        
        while True:
            try: msg = await websocket.receive_json()
            except WebSocketDisconnect: break
            mtype = msg.get("type")
            if mtype == "ping": await websocket.send_json({"type": "pong"}); continue
            elif mtype == "change_lang":
                my_lang = msg.get("lang", "ar")
                await room_manager.update_user_lang(room_code, websocket, my_lang)
                continue
            elif mtype in ["rtc-offer", "rtc-answer", "rtc-ice"]:
                await room_manager.broadcast(room_code, msg, exclude=websocket)
                continue
            elif mtype == "message":
                text = (msg.get("text") or "").strip()[:2000]
                if not text: continue
                ws_users = await room_manager.get_ws_users(room_code)
                unique_langs = set(u["lang"] for u in ws_users.values())
                
                translations = {}
                for lc in unique_langs: 
                    translations[lc] = text if lc == my_lang else text
                
                await room_manager.add_history(room_code, {"from": my_name, "original": text, "from_lang": my_lang, "translations": translations})
                for ws, u in ws_users.items():
                    payload = {"type": "translated", "from_name": my_name, "from_lang": my_lang, "original": text, "translated": text, "target_lang": u["lang"], "is_mine": (ws == websocket)}
                    try: await ws.send_json(payload)
                    except Exception: pass
    except WebSocketDisconnect: pass
    finally:
        if joined:
            await room_manager.leave(room_code, websocket)
            await room_manager.broadcast(room_code, {"type": "user_left", "name": my_name})
