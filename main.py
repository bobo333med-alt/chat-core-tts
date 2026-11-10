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

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("chat-core")

# ---------- LiveKit ----------
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "wss://chat-core-ozzy0gzk.livekit.cloud")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")

# ---------- RoomManager ----------
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

    async def cleanup(self, max_age=86400):
        now = time.time()
        async with self.lock:
            to_delete = [
                code for code, room in self.rooms.items()
                if not room["users"] or (now - room["created_at"]) > max_age
            ]
            for code in to_delete:
                del self.rooms[code]

room_manager = RoomManager()

@asynccontextmanager
async def lifespan(app):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        logger.error("⚠️ LIVEKIT_API_KEY or LIVEKIT_API_SECRET is missing!")
    else:
        logger.info("✅ LiveKit credentials found.")

    async def cleaner():
        while True:
            await asyncio.sleep(3600)
            try:
                await room_manager.cleanup()
                _cleanup_sessions()
            except Exception as e:
                logger.warning("Cleanup error: %s", e)
    task = asyncio.create_task(cleaner())
    yield
    task.cancel()

app = FastAPI(title="Chat-Core API", version="10.0", lifespan=lifespan)

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

# ---------- Providers ----------
OXEN_KEY = os.environ.get("OXEN_API_KEY", "").strip()
GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
USE_OXEN = os.environ.get("USE_OXEN", "1") == "1" and OXEN_KEY != ""

OXEN_BASE = "https://hub.oxen.ai/api/ai"
GROQ_BASE = "https://api.groq.com/openai/v1"

OXEN_TEXT_MODELS = ["claude-sonnet-4-6", "gemini-3-1-flash-lite-preview"]
OXEN_VISION_MODELS = ["qwen/qwen3.8-27b"]
GROQ_TEXT_MODELS = ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
GROQ_VISION_MODELS = ["qwen/qwen3.8-27b"]

TTS_ENABLED = os.environ.get("TTS_ENABLED", "0") == "1"
OXEN_TTS_MODEL = "bytedance-seed-audio-1-0"
TTS_VOICES = {"man": "Fritz", "woman": "Celeste", "girl": "Arista", "child": "Cillian"}

# ---------- Sessions (with TTL) ----------
_sessions: Dict[str, dict] = {}
SESSION_MAX_MSGS = 16
SESSION_TTL_SECONDS = 3600

def _cleanup_sessions():
    now = time.time()
    to_delete = [sid for sid, s in _sessions.items() if (now - s["last_at"]) > SESSION_TTL_SECONDS]
    for sid in to_delete:
        del _sessions[sid]
    if to_delete:
        logger.info(f"Sessions cleaned: {len(to_delete)}, remaining: {len(_sessions)}")

def append_session(sid, user_text, reply):
    if not sid: return
    entry = _sessions.get(sid, {"msgs": [], "last_at": time.time()})
    entry["msgs"].append({"role": "user", "content": (user_text or "")[:1000]})
    entry["msgs"].append({"role": "assistant", "content": (reply or "")[:2000]})
    entry["msgs"] = entry["msgs"][-SESSION_MAX_MSGS:]
    entry["last_at"] = time.time()
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
    target_lang: Optional[str] = None

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    lang: str = "ar"
    voice: str = "woman"
    session_id: Optional[str] = None

class MeetingTokenRequest(BaseModel):
    room_name: str
    participant_name: str

_groq_idx = {"i": 0}
def next_groq_key():
    if not GROQ_KEYS: raise HTTPException(503, detail="لا يوجد مفتاح Groq.")
    k = GROQ_KEYS[_groq_idx["i"] % len(GROQ_KEYS)]
    _groq_idx["i"] += 1
    return k

async def call_oxen(messages, models, temperature=0.7, max_tokens=2000):
    last_err = "no model"
    for model in models:
        client = AsyncOpenAI(base_url=OXEN_BASE, api_key=OXEN_KEY, timeout=60.0)
        try:
            r = await client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err = f"{model}: {str(e)[:160]}"
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
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            )
            return r.choices[0].message.content
        except Exception as e:
            last_err = f"{model}: {str(e)[:160]}"
            if "429" in str(e): await asyncio.sleep(0.6)
    raise Exception(last_err)

async def smart_chat(messages, is_vision, temperature, max_tokens):
    if USE_OXEN:
        try:
            models = OXEN_VISION_MODELS if is_vision else OXEN_TEXT_MODELS
            return await call_oxen(messages, models, temperature, max_tokens)
        except Exception as e:
            logger.warning("Oxen failed: %s", e)
    if GROQ_KEYS:
        models = GROQ_VISION_MODELS if is_vision else GROQ_TEXT_MODELS
        return await call_groq(messages, models, temperature, max_tokens)
    raise HTTPException(502, detail="لا يوجد مزوّد متاح")

LANG_NAMES = {
    "ar": "Arabic", "fr": "French", "en": "English",
    "de": "German", "tr": "Turkish", "it": "Italian",
    "zh": "Chinese", "ru": "Russian", "es": "Spanish",
    "pt": "Portuguese", "nl": "Dutch", "no": "Norwegian"
}

WHISPER_PROMPTS = {
    "ar": "محادثة طبيعية بالعربية. كلمات شائعة: نعم، لا، مرحبا، شكرا، كيف حالك، بخير، سلام، صباح الخير، مساء الخير، الحمد لله.",
    "fr": "Conversation naturelle en français. Mots courants: oui, non, bonjour, merci, comment allez-vous, ça va, salut, bonne nuit.",
    "en": "Natural English conversation. Common words: yes, no, hello, thank you, how are you, fine, good morning, good night.",
    "de": "Natürliche Unterhaltung auf Deutsch. Häufige Wörter: ja, nein, hallo, danke, wie geht es dir, gut, guten Morgen.",
    "tr": "Doğal Türkçe sohbet. Yaygın kelimeler: evet, hayır, merhaba, teşekkürler, nasılsın, iyi, günaydın.",
    "it": "Conversazione naturale in italiano. Parole comuni: sì, no, ciao, grazie, come stai, bene, buongiorno.",
    "zh": "自然的普通话对话。常用词：是，不是，你好，谢谢，你好吗，好，早上好。",
    "ru": "Естественный русский разговор. Частые слова: да, нет, привет, спасибо, как дела, хорошо, доброе утро.",
    "es": "Conversación natural en español. Palabras comunes: sí, no, hola, gracias, cómo estás, bien, buenos días.",
    "pt": "Conversa natural em português. Palavras comuns: sim, não, olá, obrigado, como vai, bem, bom dia.",
    "nl": "Natuurlijk Nederlands gesprek. Veelvoorkomende woorden: ja, nee, hallo, dank je, hoe gaat het, goed, goedemorgen.",
    "no": "Naturlig norsk samtale. Vanlige ord: ja, nei, hei, takk, hvordan har du det, bra, god morgen.",
}

async def translate_for_room(text, source_lang, target_lang):
    if source_lang == target_lang or not text.strip(): return text
    src = LANG_NAMES.get(source_lang, "the source language")
    tgt = LANG_NAMES.get(target_lang, "the target language")
    messages = [
        {"role": "system", "content": "You are a translation engine. Output ONLY the translated text, nothing else."},
        {"role": "user", "content": f"Translate from {src} to {tgt}:\n\n{text}"}
    ]
    try:
        return (await smart_chat(messages, False, 0.0, 1000)).strip()
    except Exception:
        return text

@app.get("/health")
async def health():
    return {
        "ok": True,
        "provider": "oxen" if USE_OXEN else "groq",
        "active_rooms": len(room_manager.rooms),
        "active_sessions": len(_sessions),
        "livekit_configured": bool(LIVEKIT_API_KEY and LIVEKIT_API_SECRET)
    }

@app.post("/rooms/create")
async def create_room_endpoint():
    return {"code": await room_manager.create_room()}

@app.post("/get-token")
async def get_token_endpoint(req: MeetingTokenRequest):
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        logger.error("get-token failed: LiveKit keys missing")
        raise HTTPException(500, detail="LiveKit keys not configured")
    unique_identity = f"{req.participant_name}#{secrets.token_hex(3)}"
    try:
        token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(unique_identity) \
            .with_name(req.participant_name) \
            .with_grants(api.VideoGrants(
                room_join=True, room=req.room_name,
                can_publish=True, can_subscribe=True, can_publish_data=True
            ))
        return {"token": token.to_jwt(), "url": LIVEKIT_URL}
    except Exception as e:
        logger.error(f"get-token error: {e}")
        raise HTTPException(500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.messages: raise HTTPException(400, detail="messages فارغة")
    has_image = any(
        isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"])
        for m in req.messages
    )
    if req.is_vision and not has_image: raise HTTPException(400, detail="is_vision=true بدون صورة")

    messages = req.messages
    if req.session_id and not req.is_translation:
        ctx = get_session(req.session_id)
        if ctx and messages and messages[0].get("role") == "system":
            messages = [messages[0]] + ctx + messages[1:]

    temperature = 0.0 if req.is_translation else 0.7
    max_tokens = 800 if req.is_translation else 2000

    try:
        reply = await smart_chat(messages, req.is_vision, temperature, max_tokens)
    except Exception:
        raise HTTPException(502, detail="تعذّر الحصول على رد.")

    if req.session_id and not req.is_translation:
        last_user = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
        append_session(req.session_id, last_user if isinstance(last_user, str) else "[media]", reply)

    return {"reply": reply}

@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    if not TTS_ENABLED or not USE_OXEN: raise HTTPException(503, detail="TTS غير مفعّل")
    voice = TTS_VOICES.get(req.voice, "Celeste")
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            r = await http.post(
                f"{OXEN_BASE}/audio/generate",
                headers={"Authorization": f"Bearer {OXEN_KEY}"},
                json={"model": OXEN_TTS_MODEL, "prompt": req.text[:4000], "voice": voice, "output_format": "mp3"},
            )
            if r.status_code != 200: raise HTTPException(502, detail="فشل توليد الصوت")
            return Response(content=r.content, media_type="audio/mpeg")
    except httpx.HTTPError:
        raise HTTPException(502, detail="خطأ شبكة TTS")
        # ============================================================
# /transcribe — تحصين شامل ضد الهلوسة
# ============================================================
async def _run_whisper_verbose(temp_path: str, language: str) -> dict:
    """Whisper بإعدادات صارمة (verbose_json) — لعرض الثقة."""
    key = next_groq_key()
    client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=120.0)
    lang_code = language if language in LANG_NAMES else "ar"
    prompt = WHISPER_PROMPTS.get(lang_code, WHISPER_PROMPTS["en"])
    with open(temp_path, "rb") as af:
        resp = await client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=af,
            language=lang_code,
            prompt=prompt,
            temperature=0.0,
            response_format="verbose_json",
        )
    return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)

def _should_reject(result: dict, duration_ms: int) -> Tuple[bool, str]:
    """4 فلاتر ضد الهلوسة."""
    text = (result.get("text") or "").strip()
    if not text or len(text) < 2:
        return True, "empty"
    no_speech = float(result.get("no_speech_prob") or 0.0)
    if no_speech > 0.6:
        return True, f"no_speech({no_speech:.2f})"
    avg_logprob = float(result.get("avg_logprob") or 0.0)
    if avg_logprob < -1.0:
        return True, f"low_confidence({avg_logprob:.2f})"
    compression = float(result.get("compression_ratio") or 0.0)
    if compression > 2.4:
        return True, f"repetition({compression:.2f})"
    if duration_ms > 0:
        words = len(text.split())
        seconds = duration_ms / 1000.0
        if seconds > 0 and (words / seconds) > 6:
            return True, f"too_many_words({words}w/{seconds:.1f}s)"
        if seconds < 1.0 and words > 8:
            return True, f"long_text_short_audio({words}w/{seconds:.1f}s)"
    return False, "ok"

@app.post("/transcribe")
async def transcribe_endpoint(
    file: UploadFile = File(...),
    target_lang: str = Form("ar"),
    language: str = Form(""),
    duration_ms: int = Form(0),
):
    if not GROQ_KEYS:
        raise HTTPException(503, detail="مفتاح Groq مطلوب لاستخراج النص")
    contents = await file.read()
    if len(contents) < 1000:
        return {"original": "", "translated": "", "reason": "too_small"}
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        tmp.write(contents)
    try:
        lang_code = (language or target_lang or "ar").lower()
        try:
            result = await _run_whisper_verbose(temp_path, lang_code)
        except Exception as e:
            logger.warning(f"verbose_json failed, fallback to text: {e}")
            key = next_groq_key()
            client = AsyncOpenAI(base_url=GROQ_BASE, api_key=key, timeout=120.0)
            with open(temp_path, "rb") as af:
                resp = await client.audio.transcriptions.create(
                    model="whisper-large-v3", file=af, language=lang_code,
                    prompt=WHISPER_PROMPTS.get(lang_code, ""),
                    temperature=0.0, response_format="text",
                )
            text = (resp if isinstance(resp, str) else getattr(resp, "text", "")).strip()
            if not text or len(text) < 2:
                return {"original": "", "translated": "", "reason": "empty"}
            return {"original": text, "translated": "", "reason": "ok"}

        reject, reason = _should_reject(result, duration_ms)
        if reject:
            logger.info(f"Rejected: {reason}")
            return {"original": "", "translated": "", "reason": reason}

        original = (result.get("text") or "").strip()
        if not original or len(original) < 2:
            return {"original": "", "translated": "", "reason": "empty"}

        return {
            "original": original,
            "translated": "",
            "confidence": float(result.get("avg_logprob") or 0.0),
            "language": result.get("language", lang_code),
        }
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        raise HTTPException(500, detail=f"فشل استخراج النص: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass

@app.websocket("/ws/{room_code}")
async def ws_endpoint(websocket: WebSocket, room_code: str):
    room_code = room_code.upper().strip()
    if not re.match(r'^[A-Z0-9]{6}$', room_code) and not room_code.startswith("MTG"):
        if len(room_code) < 3:
            await websocket.close(code=4000)
            return
    await websocket.accept()
    joined, my_name, my_lang = False, "User", "ar"
    try:
        try:
            first = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        except asyncio.TimeoutError:
            await websocket.close(code=4001)
            return
        if first.get("type") != "join":
            await websocket.close()
            return
        my_name = (first.get("name") or "User")[:30]
        my_lang = first.get("lang", "ar")
        if my_lang not in LANG_NAMES: my_lang = "ar"
        if not await room_manager.join(room_code, websocket, my_name, my_lang):
            await websocket.send_json({"type": "error", "message": "Room not found or full"})
            await websocket.close()
            return
        joined = True
        users = await room_manager.get_users_list(room_code)
        history_raw = await room_manager.get_history(room_code)
        history_for_me = [
            {"from": h["from"], "original": h["original"], "from_lang": h["from_lang"],
             "translated": h.get("translations", {}).get(my_lang, h["original"]), "is_mine": h["from"] == my_name}
            for h in history_raw
        ]
        await websocket.send_json({"type": "joined", "users": users, "history": history_for_me})
        await room_manager.broadcast(room_code, {"type": "user_joined", "name": my_name}, exclude=websocket)
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            mtype = msg.get("type")
            if mtype == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            elif mtype == "change_lang":
                new_lang = msg.get("lang")
                if new_lang in LANG_NAMES:
                    my_lang = new_lang
                    await room_manager.update_user_lang(room_code, websocket, my_lang)
                continue
            elif mtype in ["rtc-offer", "rtc-answer", "rtc-ice"]:
                if not room_code.startswith("MTG"):
                    await room_manager.broadcast(room_code, msg, exclude=websocket)
                continue
            elif mtype == "chat_media":
                media_type = msg.get("media_type")
                media_data = msg.get("data")
                if media_data:
                    ws_users = await room_manager.get_ws_users(room_code)
                    for ws_client in ws_users:
                        if ws_client != websocket:
                            try:
                                await ws_client.send_json({
                                    "type": "chat_media", "media_type": media_type,
                                    "data": media_data, "from_name": my_name, "from_lang": my_lang
                                })
                            except Exception:
                                pass
                continue
            elif mtype == "message":
                text = (msg.get("text") or "").strip()[:2000]
                if not text: continue
                is_translation_on = msg.get("translate", True)
                ws_users = await room_manager.get_ws_users(room_code)
                if not is_translation_on:
                    await room_manager.add_history(room_code, {
                        "from": my_name, "original": text, "from_lang": my_lang,
                        "translations": {u["lang"]: text for u in ws_users.values()},
                    })
                    for ws, u in ws_users.items():
                        payload = {
                            "type": "translated", "from_name": my_name, "from_lang": my_lang,
                            "original": text, "translated": text,
                            "target_lang": u["lang"], "is_mine": (ws == websocket),
                            "no_tts": True
                        }
                        try: await ws.send_json(payload)
                        except Exception: pass
                    continue
                unique_langs = set(u["lang"] for u in ws_users.values())
                unique_langs.add(my_lang)
                translations = {}
                for lc in unique_langs:
                    translations[lc] = text if lc == my_lang else await translate_for_room(text, my_lang, lc)
                await room_manager.add_history(room_code, {
                    "from": my_name, "original": text, "from_lang": my_lang, "translations": translations,
                })
                for ws, u in ws_users.items():
                    payload = {
                        "type": "translated", "from_name": my_name, "from_lang": my_lang,
                        "original": text, "translated": translations.get(u["lang"], text),
                        "target_lang": u["lang"], "is_mine": (ws == websocket),
                    }
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        pass
    finally:
        if joined:
            await room_manager.leave(room_code, websocket)
            await room_manager.broadcast(room_code, {"type": "user_left", "name": my_name})
