import os
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

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

# 🎯 قائمة الموديلات (السيرفر يجرب الأول، وإذا طاح يمشي للي بعدو)
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
VISION_MODEL = "llama-3.2-11b-vision-preview"

class ChatRequest(BaseModel):
    messages: list
    is_vision: bool = False

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global current_key_index
    models_to_try = [VISION_MODEL] if request.is_vision else MODELS

    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        
        for model_name in models_to_try:
            try:
                response = await client.chat.completions.create(
                    model=model_name, 
                    messages=request.messages, 
                    temperature=0.2, 
                    max_tokens=1500
                )
                return {"reply": response.choices[0].message.content}
            except Exception as e:
                error_msg = str(e)
                # إذا الموديل طاح ولا ما موجودش، نجربو اللي بعدو
                if "404" in error_msg or "model_not_found" in error_msg or "decommissioned" in error_msg:
                    continue
                # إذا المفتاح خلص، نبدلو
                elif "429" in error_msg or "Rate limit" in error_msg:
                    current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                    await asyncio.sleep(1)
                    break # نخرجو من حلقة الموديلات ونعاودو بالمفتاح الجديد
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
