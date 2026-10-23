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

# 🎯 موديلات قوية للترجمة والدردشة (بدلنا qwen خاطر يهلوس)
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
VISION_MODEL = "llama-3.2-11b-vision-preview"

class ChatRequest(BaseModel):
    messages: list
    is_vision: bool = False

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global current_key_index
    model_to_use = VISION_MODEL if request.is_vision else MODELS[0]

    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        try:
            client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            response = await client.chat.completions.create(
                model=model_to_use, messages=request.messages, temperature=0.3, max_tokens=1000
            )
            return {"reply": response.choices[0].message.content}
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                await asyncio.sleep(1)
                continue
            else:
                raise HTTPException(status_code=500, detail=error_msg)
    raise HTTPException(status_code=429, detail="All API keys reached their limit.")

# 🎯 Endpoint جديد: ياخذ الصوت، يسمعو بـ Whisper، ويكشف اللغة
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    global current_key_index
    content = await file.read()
    
    for attempt in range(len(GROQ_API_KEYS)):
        api_key = GROQ_API_KEYS[current_key_index]
        try:
            client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            
            # نبعثو الصوت لـ Whisper في Groq
            transcription = await client.audio.transcriptions.create(
                file=(file.filename, content, file.content_type),
                model="whisper-large-v3-turbo",  # موديل سريع ودقيق
                response_format="verbose_json"
            )
            
            # نرجعو النص واللغة المكتشفة (مثلا 'ar' ولا 'fr')
            return {
                "text": transcription.text,
                "language": transcription.language
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Rate limit" in error_msg:
                current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
                await asyncio.sleep(1)
                continue
            else:
                raise HTTPException(status_code=500, detail=error_msg)
    raise HTTPException(status_code=429, detail="All API keys reached their limit.")
