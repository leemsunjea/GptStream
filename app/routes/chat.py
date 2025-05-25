# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory, UserPreference # UserPreference 임포트 추가
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store, search_similar_documents # search_similar_documents 추가
from openai import OpenAI
from sqlalchemy import func, select

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Global cache for default system prompt
_default_system_prompt_cache = None

@router.post("/chat/stream")
async def chat_stream(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    message = data.get("message", "")
    global _default_system_prompt_cache # Declare usage of global variable

    openai_chat_history_list = []
    user_system_prompt = "" # 사용자별 시스템 프롬프트를 저장할 변수

    async with async_session() as session:
        # 사용자별 시스템 프롬프트 조회와 이전 대화 기록 조회를 병렬로 실행
        user_pref_task = session.execute(
            select(UserPreference).where(UserPreference.user_id == x_user_id)
        )
        # 이전 대화 기록 조회 시 최근 20개로 제한하고, 시간 역순으로 가져온 뒤 다시 정순으로 변경
        previous_chats_task = session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == x_user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(20) # 최근 20개 메시지 제한
        )

        user_pref_result, previous_chats_result = await asyncio.gather(
            user_pref_task,
            previous_chats_task
        )

        user_pref = user_pref_result.scalars().first()
        if user_pref and user_pref.system_prompt:
            user_system_prompt = user_pref.system_prompt
        else:
            if _default_system_prompt_cache: # 캐시 확인
                user_system_prompt = _default_system_prompt_cache
                print(f"[DEBUG] 사용자 {x_user_id}에게 캐시된 기본 시스템 프롬프트 적용: '{user_system_prompt}'")
            else:
                # 사용자 정의 프롬프트가 없으면, 'default_system' 프롬프트를 조회
                default_prompt_result = await session.execute(
                    select(UserPreference).where(UserPreference.user_id == "default_system").order_by(UserPreference.id) # 첫번째 기본 프롬프트를 가져오기 위해 정렬
                )
                default_prompt = default_prompt_result.scalars().first()
                if default_prompt and default_prompt.system_prompt:
                    user_system_prompt = default_prompt.system_prompt
                    _default_system_prompt_cache = user_system_prompt # 캐시에 저장
                    print(f"[DEBUG] 사용자 {x_user_id}에게 DB 기본 시스템 프롬프트 적용 및 캐시 저장: '{user_system_prompt}'")
                else:
                    # DB에도 기본 프롬프트가 없으면 최후의 기본값 사용
                    user_system_prompt = "You are a helpful assistant."
                    _default_system_prompt_cache = user_system_prompt # 캐시에 저장
                    print(f"[DEBUG] 사용자 {x_user_id}에게 최후의 기본 시스템 프롬프트 적용 및 캐시 저장: '{user_system_prompt}'")

        db_previous_chats = previous_chats_result.scalars().all()
        db_previous_chats.reverse() # OpenAI에 전달하기 위해 다시 시간 순으로 정렬

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
        # search_similar_documents 함수에 x_user_id 전달
        referenced_docs = await search_similar_documents(message, x_user_id)
        if referenced_docs:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서 수: {len(referenced_docs)}")
            if len(referenced_docs) > 0:
                print(f"[DEBUG] 사용자 {x_user_id}의 첫 번째 검색된 문서 내용: {referenced_docs[0][:100]}...") # 로그 개선
            context_text = "\n\n".join(referenced_docs)
        else:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 관련 문서를 찾지 못했습니다.")
    else:
        print(f"[DEBUG] 인덱스에 문서가 없습니다. 사용자 {x_user_id}에 대한 검색을 건너뜁니다.")
    # context_text = "\n\n".join(referenced_docs) if referenced_docs else "" # 이미 위에서 처리됨

    # 4. 시스템 프롬프트 설정 (사용자별 프롬프트 사용)
    # context_text = "\\n\\n".join(referenced_docs) if referenced_docs else "" # 이미 위에서 처리됨

    # 전역 new_system_prompt 대신 user_system_prompt 사용
    if context_text or user_system_prompt: # new_system_prompt 대신 user_system_prompt 사용
        system_prompt = (
            "다음은 사용자가 업로드한 문서에서 검색된 내용입니다. 이 내용을 기반으로 사용자의 질문에 답변해주세요. "
            "만약 내용이 질문에 답변하기에 충분하지 않다면, 그 사실을 명시하세요. "
            "또한 답변에 사용된 문서의 특정 부분을 반드시 언급하세요.\\n\\n"
            "문서 내용:\\n" + context_text + "\\n\\n"
            "\"" + user_system_prompt + "\"\\n\\n" # new_system_prompt 대신 user_system_prompt 사용
            "답변에서 줄바꿈은 '\\\\n'으로 표시하세요."
        )
    else:
        system_prompt = (
            "업로드된 문서가 없으니 일반 챗봇처럼 답변해주세요. "
            "답변에서 줄바꿈은 '\\\\n'으로 표시하세요.\\n"
            # "문서 내용:\\n" + context_text + "\\n\\n" # context_text는 비어있으므로 제거 가능
            "추가된 시스템 프롬프트:\\n" + user_system_prompt + "\\n\\n" # new_system_prompt 대신 user_system_prompt 사용
        )

    messages_to_send_to_openai = [{"role": "system", "content": system_prompt}] + openai_chat_history_list

    async def event_stream():
        full_response_content = ""
        try:
            openai_response_stream = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 모델을 "gpt-4"에서 "gpt-3.5-turbo"로 변경
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
async def add_prompt(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")): # x_user_id 추가
    data = await request.json()
    new_prompt_text = data.get("prompt", "") # 변수명 변경 new_prompt -> new_prompt_text

    if not new_prompt_text: # 변수명 변경 new_prompt -> new_prompt_text
        return {"error": "프롬프트가 비어 있습니다."}

    async with async_session() as session:
        # 기존 사용자 프롬프트 확인
        user_pref_result = await session.execute(
            select(UserPreference).where(UserPreference.user_id == x_user_id)
        )
        user_pref = user_pref_result.scalars().first()

        if user_pref:
            # 기존 프롬프트 업데이트
            previous_prompt = user_pref.system_prompt
            user_pref.system_prompt = new_prompt_text
            user_pref.updated_at = func.now() # 업데이트 시간 기록
            print(f"[DEBUG] 사용자 {x_user_id}의 프롬프트 업데이트: '{previous_prompt}' -> '{new_prompt_text}'")
        else:
            # 새 프롬프트 생성
            user_pref = UserPreference(user_id=x_user_id, system_prompt=new_prompt_text)
            session.add(user_pref)
            print(f"[DEBUG] 사용자 {x_user_id}의 새 프롬프트 생성: '{new_prompt_text}'")
        
        await session.commit()
        await session.refresh(user_pref) # DB에서 최신 정보로 객체 업데이트

    return {"success": True, "new_system_prompt": user_pref.system_prompt} # 반환값 키 변경 및 값 수정