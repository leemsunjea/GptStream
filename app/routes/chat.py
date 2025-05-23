# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store  # vector 연동
from openai import OpenAI

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    message = data.get("message", "")

    # 🔍 질문을 벡터화하고 관련 문단 검색
    context_text = ""
    referenced_docs = []
    if index.ntotal > 0:
        query_embedding = await get_embedding(message)
        D, I = index.search(np.array([query_embedding]), k=7)  # 상위 3개 문서 검색
        if I is not None and len(I[0]) > 0:
            referenced_docs = [doc_store[i] for i in I[0] if i >= 0 and i < len(doc_store)]
        if referenced_docs:
            context_text = "\n\n".join(referenced_docs)
            # 토큰 길이 제한 (예: 4000자)
            if len(context_text) > 4000:
                context_text = context_text[:4000] + "\n\n[문서 일부 생략됨]"

    # 개선된 시스템 프롬프트
    if context_text:
        system_prompt = (
            "다음은 사용자가 업로드한 문서에서 검색된 내용입니다. 이 내용을 기반으로 사용자의 질문에 답변해주세요. "
            "만약 내용이 질문에 답변하기에 충분하지 않다면, 그 사실을 명시하세요. "
            "또한 답변에 사용된 문서의 특정 부분을 반드시 언급하세요.\n\n"
            "문서 내용:\n" + context_text + "\n\n"
            "답변에서 줄바꿈은 '\n'으로 표시하세요."
        )
    else:
        system_prompt = (
            "업로드된 문서가 없으니 일반 챗봇처럼 답변해주세요. "
            "답변에서 줄바꿈은 '\n'으로 표시하세요."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    async def event_stream():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4",  # 올바른 모델 이름으로 수정
                messages=messages,
                stream=True
            )
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)

            # 참조 문서 출력
            if referenced_docs:
                yield f"\n\ndata: [참고한 문단]\n\n"
                for idx, doc in enumerate(referenced_docs, 1):
                    yield f"data: [문단 {idx}]\n{doc}\n\n"

            yield "data: \n\n[DONE]\n\n"

            # DB 저장
            async with async_session() as session:
                chat = ChatHistory(
                    user_message=message,
                    bot_response=full_response
                )
                session.add(chat)
                await session.commit()
        except Exception as e:
            import traceback
            print("오류 발생:", e)
            traceback.print_exc()
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")