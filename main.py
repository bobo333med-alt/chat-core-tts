import os
import re
import time
import asyncio
import secrets
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-core")

# ---------- Room Manager ----------
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
                return False
            if len(self.rooms[code]["users"]) >= 4:
                return False
            self.rooms[code]["users"][ws] = {"name": name, "lang": lang}
            return True

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

    async def cleanup(self, max_age=86400):
        now = time.time()
        async with self.lock:
            to_delete = [
                code for code, room in self.rooms.items()
                if not room["users"] or (now - room["created_at"]) > max_age
            ]
            for code in to_delete:
                del self.rooms[code]
            if to_delete:
                logger.info("Cleaned %d old rooms", len(to_delete))

room_manager = RoomManager()

# ---------- Lifespan (background cleanup) ----------
@asynccontextmanager
async def lifespan(app):
    async def cleaner():
        while True:
            await asyncio.sleep(3600)
            try:
                await room_manager.cleanup()
            except Exception as e:
                logger.warning("Cleanup error: %s", e)
    task = asyncio.create_task(cleaner())
    yield
    task.cancel()

app = FastAPI(title="Chat-Core API", version="4.0", lifespan=lifespan)

# ---------- CORS ----------
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS", "https://bobo333med-alt.github.io"
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# ---------- Keys ----------
OXEN_KEY = os.environ.get("OXEN_API_KEY", "").strip()
GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
USE_OXEN = os.environ.get("USE_OXEN", "1") == "1" and OXEN_KEY != ""

OXEN_BASE = "https://hub.oxen.ai/api/ai"
GROQ_BASE = "https://api.groq.com/openai/v1"

OXEN_TEXT_MODELS = ["claude-sonnet-4-6", "gemini-3-1-flash-lite-preview"]
OXEN_VISION_MODELS = ["qwen/qwen3.8-27b"]
GROQ_TEXT_MODELS = ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
GROQ_VISION_MODELS = ["qwen/qwen3.8-27b"]

# ---------- TTS ----------
TTS_ENABLED = os.environ.get("TTS_ENABLED", "0") == "1"
OXEN_TTS_MODEL = "bytedance-seed-audio-1-0"
TTS_VOICES = {"man": "Fritz", "woman": "Celeste", "girl": "Arista", "child": "Cillian"}

# ---------- Session memory ----------
_sessions: dict = {}
SESSION_MAX_MSGS = 16

def append_session(sid, user_text, reply):
    if not sid:
        return
    h = _sessions.get(sid, [])
    h.append({"role": "user", "content": (user_text or "")[:1000]})
    h.append({"role": "assistant", "content": (reply or "")[:2000]})
    _sessions[sid] = h[-SESSION_MAX_MSGS:]

def get_session(sid, last_n=6):
    if not sid:
        return []
    return _sessions.get(sid, [])[-last_n:]

# ---------- Pydantic ----------
class ChatRequest(BaseModel):
    messages: List[dict]
    is_vision: bool = False
    is_translation: bool = False
    session_id: Optional[str] = None
    target_lang: Optional[str] = None

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    lang: str = "ar"
    voice: str = "woman"
    session_id: Optional[str] = None

# ---------- Key rotation ----------
_groq_idx = {"i": 0}
def next_groq_key():
    if not GROQ_KEYS:
        raise HTTPException(503, detail="لا يوجد مفتاح Groq احتياطي.")
    k = GROQ_KEYS[_groq_idx["i"] % len(GROQ_KEYS)]
    _groq_idx["i"] += 1
    return k

# ---------- AI calls ----------
async def call_oxen(messages, models, temperature=0.7, max_tokens=2000):
    last_err = "no model"
    for model in models:
        client = AsyncOpenAI(base_url=OXEN_BASE, api_key=OXEN_KEY, timeout=60.0)
        try:
            r = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            logger.info("✓ Oxen: %s", model)
            return r.choices[0].message.content
        except Exception as e:
            last_err = f"{model}: {str(e)[:160]}"
            logger.warning("✗ Oxen %s → %s", model, last_err)
    raise Exception(last_err)

async def call_groq(messages, models, temperature=0.7, max_tokens=2000):
    last_err = "no model"
    total = len(GROQ_KEYS) * len(models) if GROQ_KEYS else 0
    for i in range(total):
        key = next_groq_key()
        model = models[i % len(models)]
        client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=60.0)
        try:
            r = await client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            logger.info("✓ Groq: %s", model)
            return r.choices[0].message.content
        except Exception as e:
            last_err = f"{model}: {str(e)[:160]}"
            logger.warning("✗ Groq %s → %s", model, last_err)
            if "429" in str(e):
                await asyncio.sleep(0.6)
    raise Exception(last_err)

async def smart_chat(messages, is_vision, temperature, max_tokens):
    if USE_OXEN:
        try:
            models = OXEN_VISION_MODELS if is_vision else OXEN_TEXT_MODELS
            return await call_oxen(messages, models, temperature, max_tokens)
        except Exception as e:
            logger.warning("Oxen failed, fallback to Groq: %s", e)
    if GROQ_KEYS:
        models = GROQ_VISION_MODELS if is_vision else GROQ_TEXT_MODELS
        return await call_groq(messages, models, temperature, max_tokens)
    raise HTTPException(502, detail="لا يوجد مزوّد متاح")

LANG_NAMES = {"ar": "Arabic", "fr": "French", "en": "English"}

async def translate_for_room(text, source_lang, target_lang):
    if source_lang == target_lang or not text.strip():
        return text
    src = LANG_NAMES.get(source_lang, "the source language")
    tgt = LANG_NAMES.get(target_lang, "the target language")
    messages = [
        {"role": "system", "content": "You are a translation engine. Output ONLY the translated text, nothing else."},
        {"role": "user", "content": f"Translate from {src} to {tgt}:\n\n{text}"}
    ]
    try:
        return (await smart_chat(messages, False, 0.0, 500)).strip()
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return text

# ---------- REST Endpoints ----------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "provider": "oxen" if USE_OXEN else "groq",
        "oxen_configured": bool(OXEN_KEY),
        "groq_keys": len(GROQ_KEYS),
        "tts": TTS_ENABLED,
        "active_rooms": len(room_manager.rooms),
    }

@app.post("/rooms/create")
async def create_room_endpoint():
    code = await room_manager.create_room()
    return {"code": code}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, detail="messages فارغة")

    has_image = any(
        isinstance(m.get("content"), list)
        and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"])
        for m in req.messages
    )
    if req.is_vision and not has_image:
        raise HTTPException(400, detail="is_vision=true بدون صورة")

    messages = req.messages
    if req.session_id and not req.is_translation:
        ctx = get_session(req.session_id)
        if ctx and messages and messages[0].get("role") == "system":
            messages = [messages[0]] + ctx + messages[1:]

    temperature = 0.0 if req.is_translation else 0.7
    max_tokens = 800 if req.is_translation else 2000

    try:
        reply = await smart_chat(messages, req.is_vision, temperature, max_tokens)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(502, detail="تعذّر الحصول على رد. أعد المحاولة.")

    if req.session_id and not req.is_translation:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                last_user = c if isinstance(c, str) else "[media]"
                break
        append_session(req.session_id, last_user, reply)

    return {"reply": reply}

@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    if not TTS_ENABLED:
        raise HTTPException(503, detail="TTS غير مفعّل على السيرفر")
    if not USE_OXEN:
        raise HTTPException(503, detail="TTS يتطلب Oxen مفعّلاً")
    if req.lang not in ("en", "ar", "fr"):
        raise HTTPException(400, detail="lang غير مدعوم")

    voice = TTS_VOICES.get(req.voice, "Celeste")
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(
                f"{OXEN_BASE}/audio/generate",
                headers={"Authorization": f"Bearer {OXEN_KEY}"},
                json={
                    "model": OXEN_TTS_MODEL,
                    "prompt": req.text[:4000],
                    "voice": voice,
                    "output_format": "mp3",
                },
            )
            if r.status_code != 200:
                logger.error("TTS %s: %s", r.status_code, r.text[:200])
                raise HTTPException(502, detail="فشل توليد الصوت")
            return Response(content=r.content, media_type="audio/mpeg")
    except httpx.HTTPError as e:
        raise HTTPException(502, detail=f"خطأ شبكة TTS: {type(e).__name__}")

# ---------- WebSocket ----------
@app.websocket("/ws/{room_code}")
async def ws_endpoint(websocket: WebSocket, room_code: str):
    room_code = room_code.upper().strip()
    if not re.match(r'^[A-Z0-9]{6}$', room_code):
        await websocket.close(code=4000)
        return

    await websocket.accept()
    joined = False
    my_name = "User"
    my_lang = "ar"

    try:
        # First message must be join
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        except asyncio.TimeoutError:
            await websocket.close(code=4001)
            return

        if first.get("type") != "join":
            await websocket.send_json({"type": "error", "message": "Expected join"})
            await websocket.close()
            return

        my_name = (first.get("name") or "User")[:30]
        my_lang = first.get("lang", "ar")
        if my_lang not in ("ar", "fr", "en"):
            my_lang = "ar"

        ok = await room_manager.join(room_code, websocket, my_name, my_lang)
        if not ok:
            await websocket.send_json({"type": "error", "message": "Room not found or full"})
            await websocket.close()
            return
        joined = True
        logger.info("WS joined: %s (%s) room=%s", my_name, my_lang, room_code)

        # Send joined confirmation + history in joiner's lang
        users = await room_manager.get_users_list(room_code)
        history_raw = await room_manager.get_history(room_code)
        history_for_me = [
            {
                "from": h["from"],
                "original": h["original"],
                "from_lang": h["from_lang"],
                "translated": h.get("translations", {}).get(my_lang, h["original"]),
                "is_mine": h["from"] == my_name,
            }
            for h in history_raw
        ]
        await websocket.send_json({
            "type": "joined",
            "users": users,
            "history": history_for_me,
        })

        await room_manager.broadcast(
            room_code,
            {"type": "user_joined", "name": my_name},
            exclude=websocket,
        )

        # Message loop
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            mtype = msg.get("type")

            if mtype == "ping":
                try:
                    await websocket.send_json({"type": "pong"})
                except Exception:
                    break
                continue

            if mtype == "message":
                text = (msg.get("text") or "").strip()[:2000]
                if not text:
                    continue

                ws_users = await room_manager.get_ws_users(room_code)
                unique_langs = set(u["lang"] for u in ws_users.values())
                unique_langs.add(my_lang)

                # Compute translations once per unique lang
                translations = {}
                for lc in unique_langs:
                    if lc == my_lang:
                        translations[lc] = text
                    else:
                        translations[lc] = await translate_for_room(text, my_lang, lc)

                # Save to history
                await room_manager.add_history(room_code, {
                    "from": my_name,
                    "original": text,
                    "from_lang": my_lang,
                    "translations": translations,
                })

                # Send to each user
                for ws, u in ws_users.items():
                    translated = translations.get(u["lang"], text)
                    payload = {
                        "type": "translated",
                        "from_name": my_name,
                        "from_lang": my_lang,
                        "original": text,
                        "translated": translated,
                        "target_lang": u["lang"],
                        "is_mine": (ws == websocket),
                    }
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WS error: %s", e)
    finally:
        if joined:
            await room_manager.leave(room_code, websocket)
            await room_manager.broadcast(room_code, {"type": "user_left", "name": my_name})
            logger.info("WS left: %s room=%s", my_name, room_code)
