import os
import asyncio
from fastapi import FastAPI, HTTPException
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

# 🎯 هنا السيرفر ياخذ المفاتيح من إعدادات Render (موش من الكود)
# تقدر تحط أكثر من مفتاح في Render وتفصل بيناتهم بفاصلة (,)
keys_env = os.environ.get("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in keys_env.split(",") if k.strip()]

# في حالة ما لقاش مفاتيح، نعطيو مفتاح وهمي باش ما يطيحش السيرفر
if not GROQ_API_KEYS:
    GROQ_API_KEYS = ["gsk_dummy_key_please_set_env_var"]

current_key_index = 0

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
VISION_MODEL = "qwen/qwen3.8-27b"

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
                model=model_to_use, messages=request.messages, temperature=0.4, max_tokens=1000
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
