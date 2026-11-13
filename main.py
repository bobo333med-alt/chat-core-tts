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
        logger.error("⚠️ LiveKit keys missing!")
    if not GROQ_KEYS:
        logger.error("⚠️ GROQ_API_KEYS empty!")
    else:
        logger.info(f"✅ {len(GROQ_KEYS)} Groq key(s) loaded.")
    yield

app = FastAPI(title="Chat-Core API", version="21.4 Ultra", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
GROQ_BASE = "https://api.groq.com/openai/v1"

GROQ_TEXT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
GROQ_FAST_MODEL = "openai/gpt-oss-20b"

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

class TTSRequest(BaseModel):
    text: str
    lang: str = "ar"
    voice: str = "woman"
    session_id: Optional[str] = None

_groq_idx = {"i": 0}
def next_groq_key():
    if not GROQ_KEYS: raise HTTPException(503, detail="لا يوجد مفتاح Groq.")
    k = GROQ_KEYS[_groq_idx["i"] % len(GROQ_KEYS)]
    _groq_idx["i"] += 1
    return k

def _classify_groq_error(e: Exception) -> Tuple[int, str]:
    status = getattr(e, "status_code", None)
    msg = str(e)
    if status == 401 or "invalid_api_key" in msg: return 401, "مفتاح Groq غير صالح."
    if status == 429 or "rate_limit" in msg: return 429, "تجاوز حد Groq مؤقتاً."
    return 502, f"خطأ Groq: {msg[:200]}"

async def call_groq(messages, models, temperature=0.7, max_tokens=2000, timeout=30.0):
    last_err = "no model tried"
    if not GROQ_KEYS: raise HTTPException(503, detail="No GROQ_KEYS.")
    for model in models:
        key = next_groq_key()
        client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=timeout)
        try:
            r = await client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            last_err = str(e)
            if "rate_limit" in last_err or "429" in last_err: await asyncio.sleep(0.8)
            continue
    raise HTTPException(*_classify_groq_error(Exception(last_err)))

async def smart_chat(messages, is_vision, temperature, max_tokens):
    return await call_groq(messages, GROQ_TEXT_MODELS, temperature, max_tokens)

LANG_NAMES = {"ar": "Arabic", "fr": "French", "en": "English", "de": "German", "tr": "Turkish", "it": "Italian", "zh": "Chinese", "ru": "Russian", "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "no": "Norwegian"}

_trans_cache: Dict[str, Tuple[str, float]] = {}
CACHE_TTL = 3600

def _cache_key(text: str, src: str, tgt: str) -> str:
    return f"{src}:{tgt}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"

async def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not text.strip() or src_lang == tgt_lang: return text
    k = _cache_key(text, src_lang, tgt_lang)
    item = _trans_cache.get(k)
    if item and time.time() - item[1] < CACHE_TTL: return item[0]
    try:
        result = await call_groq(
            messages=[
                {"role": "system", "content": f"Translate from {LANG_NAMES.get(src_lang, src_lang)} to {LANG_NAMES.get(tgt_lang, tgt_lang)}. Output ONLY the translation."},
                {"role": "user", "content": text},
            ],
            models=[GROQ_FAST_MODEL, "openai/gpt-oss-120b"], temperature=0.0, max_tokens=800, timeout=20.0
        )
        out = (result or "").strip().strip('"').strip("'")
        if out:
            _trans_cache[k] = (out, time.time())
            return out
        return text
    except Exception:
        return text

async def translate_text_chunked(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not text.strip() or src_lang == tgt_lang: return text
    if len(text) < 1000:
        return await translate_text(text, src_lang, tgt_lang)
    chunks = []
    current_chunk = ""
    for sentence in text.replace('\n', ' ').split('. '):
        if len(current_chunk) + len(sentence) < 800:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip(): chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk.strip(): chunks.append(current_chunk.strip())
    translated_chunks = []
    for chunk in chunks:
        translated_chunks.append(await translate_text(chunk, src_lang, tgt_lang))
    return " ".join(translated_chunks)

HALLUCINATION_BLACKLIST = [
    "كلمات شائعة", "شائعة تشير", "تخيلها", "ترمزية", "شائعة بالعربية", 
    "الموضوعات في الوصف", "اشتركوا في القناة", "شكراً لكم", "شكرا لكم",
    "Thank you for watching", "Subscribe to", "Amara.org", "ترجمة"
]

async def _run_whisper_verbose(temp_path: str, language: str) -> dict:
    key = next_groq_key()
    client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=120.0)
    with open(temp_path, "rb") as af:
        kwargs = {"model": "whisper-large-v3", "file": af, "temperature": 0.0, "response_format": "verbose_json"}
        if language and language in LANG_NAMES:
            kwargs["language"] = language
        resp = await client.audio.transcriptions.create(**kwargs)
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

def _should_reject(result: dict, duration_ms: int) -> Tuple[bool, str]:
    text = (result.get("text") or "").strip()
    if not text or len(text) < 2: return True, "empty"
    # تم تعديل الشرط ليقبل الفيديوهات (التي ترسل duration_ms = 0)
    if duration_ms and duration_ms > 0 and duration_ms < 1000: return True, "too_short"
    for bad_word in HALLUCINATION_BLACKLIST:
        if bad_word in text: return True, "hallucination"
    no_speech = float(result.get("no_speech_prob") or 0.0)
    if no_speech > 0.4: return True, "no_speech"
    return False, "ok"

TTS_VOICES = {
    "ar": {"man": "ar-SA-HamedNeural", "woman": "ar-SA-ZariyahNeural", "girl": "ar-EG-SalmaNeural", "child": "ar-EG-SalmaNeural"},
    "fr": {"man": "fr-FR-HenriNeural", "woman": "fr-FR-DeniseNeural", "girl": "fr-FR-EloiseNeural", "child": "fr-FR-EloiseNeural"},
    "en": {"man": "en-US-GuyNeural", "woman": "en-US-JennyNeural", "girl": "en-US-AriaNeural", "child": "en-US-AnaNeural"},
    "de": {"man": "de-DE-ConradNeural", "woman": "de-DE-KatjaNeural", "girl": "de-DE-AmelieNeural", "child": "de-DE-AmelieNeural"},
    "tr": {"man": "tr-TR-AhmetNeural", "woman": "tr-TR-EmelNeural", "girl": "tr-TR-EmelNeural", "child": "tr-TR-EmelNeural"},
    "it": {"man": "it-IT-DiegoNeural", "woman": "it-IT-ElsaNeural", "girl": "it-IT-IsabellaNeural", "child": "it-IT-IsabellaNeural"},
    "zh": {"man": "zh-CN-YunxiNeural", "woman": "zh-CN-XiaoxiaoNeural", "girl": "zh-CN-XiaoyiNeural", "child": "zh-CN-XiaoyiNeural"},
    "ru": {"man": "ru-RU-DmitryNeural", "woman": "ru-RU-SvetlanaNeural", "girl": "ru-RU-DariyaNeural", "child": "ru-RU-DariyaNeural"},
    "es": {"man": "es-ES-AlvaroNeural", "woman": "es-ES-ElviraNeural", "girl": "es-ES-LuciaNeural", "child": "es-ES-LuciaNeural"},
    "pt": {"man": "pt-PT-DuarteNeural", "woman": "pt-PT-RaquelNeural", "girl": "pt-PT-RaquelNeural", "child": "pt-PT-RaquelNeural"},
    "nl": {"man": "nl-NL-MaartenNeural", "woman": "nl-NL-ColetteNeural", "girl": "nl-NL-ColetteNeural", "child": "nl-NL-ColetteNeural"},
    "no": {"man": "nb-NO-FinnNeural", "woman": "nb-NO-PernilleNeural", "girl": "nb-NO-PernilleNeural", "child": "nb-NO-PernilleNeural"},
}

def _pick_voice(lang: str, voice: str) -> Tuple[str, str, str]:
    table = TTS_VOICES.get((lang or "ar").lower()) or TTS_VOICES["en"]
    name = table.get((voice or "woman").lower()) or table["woman"]
    return name, "+0%", "+0Hz"

async def synthesize_tts(text: str, lang: str, voice: str) -> bytes:
    name, rate, pitch = _pick_voice(lang, voice)
    buf = bytearray()
    communicate = edge_tts.Communicate(text, name, rate=rate, pitch=pitch)
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio": buf.extend(chunk["data"])
    return bytes(buf)

@app.get("/health")
async def health(): return {"ok": True, "keys": len(GROQ_KEYS)}

@app.get("/health/groq")
async def health_groq():
    if not GROQ_KEYS: return {"status": "missing"}
    try:
        await call_groq([{"role": "user", "content": "ping"}], [GROQ_FAST_MODEL], temperature=0.0, max_tokens=5, timeout=10.0)
        return {"status": "ok"}
    except Exception as e: return {"status": "error", "detail": str(e)}

@app.post("/rooms/create")
async def create_room_endpoint(): return {"code": await room_manager.create_room()}

@app.post("/get-token")
async def get_token_endpoint(req: MeetingTokenRequest):
    if not LIVEKIT_API_KEY: raise HTTPException(500, detail="LiveKit keys missing")
    uid = f"{req.participant_name}#{secrets.token_hex(3)}"
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET).with_identity(uid).with_name(req.participant_name).with_grants(api.VideoGrants(room_join=True, room=req.room_name, can_publish=True, can_subscribe=True))
    return {"token": token.to_jwt(), "url": LIVEKIT_URL}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages: raise HTTPException(400, detail="messages فارغة")
    messages = req.messages
    if req.session_id and not req.is_translation:
        ctx = get_session(req.session_id)
        if ctx and messages[0].get("role") == "system": messages = [messages[0]] + ctx + messages[1:]
    reply = await smart_chat(messages, req.is_vision, 0.0 if req.is_translation else 0.7, 800 if req.is_translation else 2000)
    if req.session_id and not req.is_translation:
        last_user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
        append_session(req.session_id, last_user if isinstance(last_user, str) else "[media]", reply)
    return {"reply": reply}

@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    text = (req.text or "").strip()
    if not text: raise HTTPException(400, detail="text فارغ")
    try:
        audio = await synthesize_tts(text[:4000], req.lang, req.voice)
        return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=3600"})
    except Exception as e: raise HTTPException(503, detail=f"TTS Error: {e}")

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...), target_lang: str = Form("ar"), language: str = Form(""), duration_ms: int = Form(0)):
    contents = await file.read()
    if len(contents) < 1000: return {"original": "", "translated": "", "reason": "too_small"}
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name; tmp.write(contents)
    try:
        result = await _run_whisper_verbose(temp_path, language)
        reject, reason = _should_reject(result, duration_ms)
        if reject: return {"original": "", "translated": "", "reason": reason}
        original = result.get("text", "").strip()
        translated = original
        if target_lang and target_lang != (language or "ar"):
            translated = await translate_text_chunked(original, language or "ar", target_lang)
        return {"original": original, "translated": translated, "reason": "ok"}
    except Exception as e: raise HTTPException(500, detail=f"Transcribe Error: {e}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.websocket("/ws/{room_code}")
async def ws_endpoint(websocket: WebSocket, room_code: str):
    room_code = room_code.upper().strip()
    await websocket.accept()
    joined, my_name, my_lang = False, "User", "ar"
    try:
        first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        if first.get("type") != "join": await websocket.close(); return
        my_name = (first.get("name") or "User")[:30]
        my_lang = first.get("lang", "ar")
        if not await room_manager.join(room_code, websocket, my_name, my_lang):
            await websocket.send_json({"type": "error", "message": "Room full"}); await websocket.close(); return
        joined = True
        await websocket.send_json({"type": "joined", "users": await room_manager.get_users_list(room_code), "history": await room_manager.get_history(room_code)})
        await room_manager.broadcast(room_code, {"type": "user_joined", "name": my_name}, exclude=websocket)
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            if mtype == "ping": await websocket.send_json({"type": "pong"})
            elif mtype == "change_lang":
                my_lang = msg.get("lang", "ar"); await room_manager.update_user_lang(room_code, websocket, my_lang)
            elif mtype in ["rtc-offer", "rtc-answer", "rtc-ice"]: await room_manager.broadcast(room_code, msg, exclude=websocket)
            elif mtype == "message":
                text = (msg.get("text") or "").strip()[:2000]
                if not text: continue
                ws_users = await room_manager.get_ws_users(room_code)
                target_langs = set(u["lang"] for u in ws_users.values()) - {my_lang}
                translations: Dict[str, str] = {my_lang: text}
                for tgt in target_langs:
                    translations[tgt] = await translate_text_chunked(text, my_lang, tgt)
                await room_manager.add_history(room_code, {"from": my_name, "original": text, "from_lang": my_lang, "translations": translations})
                for ws, u in ws_users.items():
                    try: await ws.send_json({"type": "translated", "from_name": my_name, "from_lang": my_lang, "original": text, "translated": translations.get(u["lang"], text), "target_lang": u["lang"], "is_mine": (ws == websocket)})
                    except Exception: pass
    except (WebSocketDisconnect, asyncio.TimeoutError): pass
    finally:
        if joined:
            await room_manager.leave(room_code, websocket)
            await room_manager.broadcast(room_code, {"type": "user_left", "name": my_name})
