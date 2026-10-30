import os
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from openai import AsyncOpenAI
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- إعداد المفاتيح ---
keys_env = os.environ.get("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]
if not GROQ_API_KEYS:
    GROQ_API_KEYS = ["gsk_dummy_key_please_set_env_var"]

OXEN_API_KEY = os.environ.get("OXEN_API_KEY", "")
if not OXEN_API_KEY:
    OXEN_API_KEY = "oxen_dummy_key_please_set_env_var"

current_key_index = 0

# --- إعداد النماذج ---
# نماذج Groq (كخطة احتياطية)
GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
GROQ_VISION_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct"]

# نماذج Oxen (الخيار الأساسي)
# يمكنك تغيير اسم النموذج حسب ما هو متاح في حسابك على Oxen
OXEN_CHAT_MODEL = "claude-sonnet-4-6"
OXEN_VISION_MODEL = "claude-sonnet-4-6" # نفس النموذج يدعم الرؤية

class ChatRequest(BaseModel):
    messages: list
    is_vision: bool = False
    is_translation: bool = False 

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global current_key_index
    temp = 0.0 if request.is_translation else 0.7
    
    # --- 1. المحاولة الأولى: Oxen.ai ---
    if OXEN_API_KEY and not OXEN_API_KEY.startswith("oxen_dummy"):
        try:
            oxen_client = AsyncOpenAI(
                base_url="https://hub.oxen.ai/api/ai",
                api_key=OXEN_API_KEY
            )
            model_name = OXEN_VISION_MODEL if request.is_vision else OXEN_CHAT_MODEL
            
            response = await oxen_client.chat.completions.create(
                model=model_name,
                messages=request.messages,
                temperature=temp,
                max_tokens=2000
            )
            return {"reply": response.choices[0].message.content}
        except Exception as e:
            print(f"Oxen Error: {str(e)}. Falling back to Groq...")
            # إذا فشل Oxen، نكمل إلى Groq

    # --- 2. المحاولة الثانية: Groq (خطة احتياطية) ---
    last_error = "Unknown error"
    models_to_try = GROQ_VISION_MODELS if request.is_vision else GROQ_MODELS

    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        try:
            groq_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            
            for model_name in models_to_try:
                try:
                    response = await groq_client.chat.completions.create(
                        model=model_name, 
                        messages=request.messages, 
                        temperature=temp, 
                        max_tokens=2000
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

@app.post("/tts")
async def proxy_tts(request: Request):
    try:
        data = await request.json()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://chat-core-voice.onrender.com/v1/audio/speech",
                json=data,
                timeout=60.0
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            return Response(content=response.content, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
