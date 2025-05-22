# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory
from db.database import async_session
from app.vector_db import get_embedding, index, doc_store  # vector 연동

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")

    # 🔍 질문을 벡터화하고 관련 문단 검색
    context_text = ""
    if index.ntotal > 0:
        query_embedding = get_embedding(message)
        D, I = index.search(np.array([query_embedding]), k=3)
        related_docs = []
        if I is not None and len(I[0]) > 0:
            related_docs = [doc_store[i] for i in I[0] if i >= 0 and i < len(doc_store)]
        if related_docs:
            context_text = "\n\n".join(related_docs)

    # system 프롬프트 생성
    if context_text:
        system_prompt = (
            "다음은 사용자가 업로드한 문서의 일부입니다. 해당 내용을 바탕으로 정확하고 친절하게 답변해주세요:\n\n" + context_text +
            "\n\n또한 사용자에게 문서의 어떤 부분을 참고했는지를 알려주세요.\n\n"
        )
    else:
        system_prompt = "업로드된 문서가 없으니 일반 챗봇처럼 답변해주세요."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    async def event_stream():
        full_response = ""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            )
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.get("content")
                    if content:
                        full_response += content
                        yield f"data: {content}\n\n"
                        await asyncio.sleep(0)

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