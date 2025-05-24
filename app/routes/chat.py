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
        if query_embedding is None:
            print("[ERROR] 쿼리 임베딩 생성 실패")
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        print(f"[DEBUG] FAISS 인덱스 벡터 개수: {index.ntotal}")
        try:
            D, I = index.search(np.array([query_embedding]), k=3)  # 상위 7개 문서 검색
            print(f"[DEBUG] 검색 결과 인덱스: {I}, 거리: {D}")
        except Exception as e:
            print(f"[ERROR] FAISS 검색 중 오류 발생: {e}")
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        if I is not None and len(I[0]) > 0:
            referenced_docs = [doc_store[i] for i in I[0] if i >= 0 and i < len(doc_store)]
            print(f"[DEBUG] 검색된 문서 개수: {len(referenced_docs)}")
            if len(referenced_docs) > 0:
                print(f"[DEBUG] 첫 번째 검색된 문서 내용: {referenced_docs[0]}")  # [:100] 제거
            else:
                print("[DEBUG] 검색된 문서가 없습니다.")
        else:
            print("[DEBUG] 검색 결과가 없습니다.")
    else:
        print("[DEBUG] 인덱스에 문서가 없습니다.")

    # 문서 검색 후 context_text 생성
    context_text = "\n\n".join(referenced_docs) if referenced_docs else ""

    # 개선된 시스템 프롬프트
    new_system_prompt = data.get("new_system_prompt", "")  # 사용자로부터 받은 새로운 시스템 프롬프트

    # 이전 대화 기록을 가져오기 위한 전역 변수
    global chat_history
    if 'chat_history' not in globals():
        chat_history = []

    # 이전 대화 기록 추가
    chat_history.append({"role": "user", "content": message})

    if context_text or new_system_prompt:
        system_prompt = (
            "다음은 사용자가 업로드한 문서에서 검색된 내용입니다. 이 내용을 기반으로 사용자의 질문에 답변해주세요. "
            "만약 내용이 질문에 답변하기에 충분하지 않다면, 그 사실을 명시하세요. "
            "또한 답변에 사용된 문서의 특정 부분을 반드시 언급하세요.\n\n"
            "문서 내용:\n" + context_text + "\n\n"
            "추가된 시스템 프롬프트:\n" + new_system_prompt + "\n\n"
            "답변에서 줄바꿈은 '\n'으로 표시하세요."
        )

    # 이전 대화 기록을 시스템 메시지에 추가
    messages = [{"role": "system", "content": system_prompt}] + chat_history

    async def event_stream():
        full_response = ""
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                stream=True
            )
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    full_response += content
                    yield f"data: {content}\n\n"
                    await asyncio.sleep(0)

            # 봇 응답을 대화 기록에 추가
            chat_history.append({"role": "assistant", "content": full_response})

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

@router.post("/chat/add_prompt")
async def add_prompt(request: Request):
    data = await request.json()
    new_prompt = data.get("prompt", "")

    if not new_prompt:
        return {"error": "프롬프트가 비어 있습니다."}

    # 새로운 시스템 프롬프트를 전역 변수로 저장
    global new_system_prompt
    new_system_prompt = new_prompt

    return {"success": True, "chatHistory": new_system_prompt}