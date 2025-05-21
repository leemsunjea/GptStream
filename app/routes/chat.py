# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import openai
from app.config import settings
from db.models import ChatHistory
from db.database import async_session

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")

    async def event_stream():
        full_response = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": message}],
                stream=True
            )
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.get("content")
                    if content:
                        full_response += content
                        yield f"data: {content}\n\n"
                        await asyncio.sleep(0)
            # 대화 저장
            async with async_session() as session:
                chat = ChatHistory(user_message=message, bot_response=full_response)
                session.add(chat)
                await session.commit()
            yield "data: [DONE]\n\n"
        except Exception as e:
            import traceback
            print("DB 저장 중 오류:", e)
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")