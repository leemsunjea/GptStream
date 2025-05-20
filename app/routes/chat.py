# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import openai
from app.config import settings

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")

    async def event_stream():
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.get("content")
                if content:
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)  # ⭐ flush 보장
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")