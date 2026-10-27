import os
import asyncio
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from openai import AsyncOpenAI
import edge_tts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

keys_env = os.environ.get("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
if not GROQ_API_KEYS:
    GROQ_API_KEYS = ["gsk_dummy_key_please_set_env_var"]

current_key_index = 0

MODELS = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
# 🎯 الموديل الوحيد الشغال حالياً للصور في Groq
VISION_MODELS = ["qwen/qwen3.8-27b"]

class ChatRequest(BaseModel):
    messages: list
    is_vision: bool = False
    is_translation: bool = False 

class TTSRequest(BaseModel):
    input: str
    voice: str
    rate: str = "-10%"
    pitch: str = "+0Hz"

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global current_key_index
    models_to_try = VISION_MODELS if request.is_vision else MODELS
    temp = 0.0 if request.is_translation else 0.7
    last_error = "Unknown error"
    
    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        try:
            client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            for model_name in models_to_try:
                try:
                    response = await client.chat.completions.create(
                        model=model_name, messages=request.messages, temperature=temp, max_tokens=2000
                    )
                    return {"reply": response.choices[0].message.content}
                except Exception as e:
                    last_error = f"Groq Error ({model_name}): {str(e)}"
                    continue 
            raise Exception(last_error)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                await asyncio.sleep(1)
                continue
            else:
                raise HTTPException(status_code=500, detail=error_msg)
    raise HTTPException(status_code=429, detail="All API keys reached their limit.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    global current_key_index
    content = await file.read()
    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        try:
            client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            transcription = await client.audio.transcriptions.create(
                file=(file.filename, content, file.content_type),
                model="whisper-large-v3-turbo", 
                response_format="verbose_json"
            )
            return {"text": transcription.text, "language": transcription.language}
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                await asyncio.sleep(1)
                continue
            else:
                raise HTTPException(status_code=500, detail=error_msg)
    raise HTTPException(status_code=429, detail="All API keys reached their limit.")

# 🎯 دالة الصوت المدمجة (Edge-TTS) - ما عادش نحتاجوا سيرفر منفصل
@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name
        
        communicate = edge_tts.Communicate(request.input, request.voice, rate=request.rate, pitch=request.pitch)
        await communicate.save(tmp_path)
        
        with open(tmp_path, "rb") as f:
            audio_data = f.read()
        
        os.unlink(tmp_path)
        return Response(content=audio_data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
