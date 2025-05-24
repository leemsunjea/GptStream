# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store  # vector 연동
from openai import OpenAI
from sqlalchemy import func, select

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/chat/stream")
async def chat_stream(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    message = data.get("message", "")  # 현재 사용자의 메시지

    # 1. 이전 대화 기록을 불러와 OpenAI에 전달할 형식으로 구성
    openai_chat_history_list = []
    async with async_session() as session:
        previous_chats_result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == x_user_id)
            .order_by(ChatHistory.created_at)
        )
        db_previous_chats = previous_chats_result.scalars().all()

        for chat_entry in db_previous_chats:
            openai_chat_history_list.append({"role": "user", "content": chat_entry.user_message})
            if chat_entry.bot_response:  # 봇 응답이 있는 경우에만 추가
                openai_chat_history_list.append({"role": "assistant", "content": chat_entry.bot_response})

    # 2. 현재 사용자의 메시지를 대화 기록에 추가
    openai_chat_history_list.append({"role": "user", "content": message})

    # 3. 🔍 질문을 벡터화하고 관련 문단 검색 (기존 로직 유지)
    context_text = ""
    referenced_docs = []
    if index.ntotal > 0:
        query_embedding = await get_embedding(message)
        if query_embedding is None:
            print("[ERROR] 쿼리 임베딩 생성 실패")
            # event_stream 함수를 직접 호출할 수 없으므로, 빈 스트림 응답을 위한 간단한 처리가 필요할 수 있습니다.
            # 여기서는 일단 에러 상황을 가정하고 빈 응답을 생성하는 더미 event_stream을 가정합니다.
            # 실제로는 이 부분에서 StreamingResponse를 바로 반환하거나, event_stream 내부에서 처리해야 합니다.
            async def dummy_event_stream_on_error():
                yield f"data: [ERROR] 쿼리 임베딩 생성 실패\n\n"
                yield f"data: \n\n[DONE]\n\n"
            return StreamingResponse(dummy_event_stream_on_error(), media_type="text/event-stream")

        print(f"[DEBUG] FAISS 인덱스 벡터 개수: {index.ntotal}")
        try:
            D, I = index.search(np.array([query_embedding]), k=3)
            print(f"[DEBUG] 검색 결과 인덱스: {I}, 거리: {D}")
        except Exception as e:
            print(f"[ERROR] FAISS 검색 중 오류 발생: {e}")
            async def dummy_event_stream_on_faiss_error():
                yield f"data: [ERROR] FAISS 검색 중 오류: {str(e)}\n\n"
                yield f"data: \n\n[DONE]\n\n"
            return StreamingResponse(dummy_event_stream_on_faiss_error(), media_type="text/event-stream")

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
    context_text = "\n\n".join(referenced_docs) if referenced_docs else ""

    # 4. 시스템 프롬프트 설정 (기존 로직 유지, new_system_prompt는 전역 변수)
    global new_system_prompt
    if 'new_system_prompt' not in globals():
        new_system_prompt = ""

    if context_text or new_system_prompt:
        system_prompt = (
            "다음은 사용자가 업로드한 문서에서 검색된 내용입니다. 이 내용을 기반으로 사용자의 질문에 답변해주세요. "
            "만약 내용이 질문에 답변하기에 충분하지 않다면, 그 사실을 명시하세요. "
            "또한 답변에 사용된 문서의 특정 부분을 반드시 언급하세요.\n\n"
            "문서 내용:\n" + context_text + "\n\n"
            "" + new_system_prompt + ""
            "답변에서 줄바꿈은 '\\n'으로 표시하세요."
        )
    else:
        system_prompt = (
            "업로드된 문서가 없으니 일반 챗봇처럼 답변해주세요. "
            "답변에서 줄바꿈은 '\\n'으로 표시하세요."
            "문서 내용:\n" + context_text + "\n\n" # context_text는 비어있을 수 있음
            "추가된 시스템 프롬프트:\n" + new_system_prompt + "\n\n" # new_system_prompt는 비어있을 수 있음
        )

    messages_to_send_to_openai = [{"role": "system", "content": system_prompt}] + openai_chat_history_list

    async def event_stream():
        full_response_content = ""
        try:
            openai_response_stream = client.chat.completions.create(
                model="gpt-4",
                messages=messages_to_send_to_openai,
                stream=True
            )
            for chunk in openai_response_stream:
                content_piece = getattr(chunk.choices[0].delta, "content", None)
                if content_piece:
                    full_response_content += content_piece
                    yield f"data: {content_piece}\n\n"
                    await asyncio.sleep(0) # 클라이언트 처리를 위한 약간의 지연

            # 5. 현재 사용자 메시지와 봇의 전체 응답을 DB에 하나의 레코드로 저장
            async with async_session() as session:
                new_exchange_record = ChatHistory(
                    user_id=x_user_id,
                    user_message=message,
                    bot_response=full_response_content,
                    created_at=func.now() # SQLAlchemy가 자동으로 처리하도록 server_default를 사용하거나 명시적 설정
                )
                session.add(new_exchange_record)
                await session.commit()

            if referenced_docs:
                yield f"\n\ndata: [참고한 문단]\n\n"
                for idx, doc_content in enumerate(referenced_docs, 1): # 변수명 수정 doc -> doc_content
                    yield f"data: [문단 {idx}]\n{doc_content}\n\n"

            yield "data: \n\n[DONE]\n\n"

        except Exception as e:
            import traceback
            print(f"오류 발생 in event_stream: {e}")
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
    previous_prompt = new_system_prompt if 'new_system_prompt' in globals() else "(없음)"
    new_system_prompt = new_prompt

    # 로그 출력
    print("[DEBUG] 이전 프롬프트:", previous_prompt)
    print("[DEBUG] 새로운 프롬프트:", new_system_prompt)

    return {"success": True, "chatHistory": new_system_prompt}