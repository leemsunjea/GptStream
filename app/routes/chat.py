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

# 새로운 기본 시스템 프롬프트 템플릿
BASE_SYSTEM_PROMPT_TEMPLATE = """
{user_specific_prompt_or_default}

{reference_document_section}

이제 사용자의 다음 질문에 답변해주세요. 답변은 한국어로 작성합니다.
답변 내용에 줄바꿈('\\n')은 문맥상 명확한 단락 구분이 필요하거나 목록을 나열하는 등, 반드시 필요한 경우에만 최소한으로 사용해주세요.
단어 중간이나 불필요한 위치에 줄바꿈을 사용하지 마세요. 간결하고 읽기 쉽게 답변해주세요.
"""

@router.post("/chat/stream")
async def chat_stream(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    message = data.get("message", "")
    global _default_system_prompt_cache

    openai_chat_history_list = []
    user_system_prompt_content = "" # 사용자가 설정한 프롬프트 또는 기본 프롬프트 내용

    async with async_session() as session:
        # 1. 사용자 시스템 프롬프트 또는 기본 시스템 프롬프트 가져오기
        user_pref_result = await session.execute(
            select(UserPreference).where(UserPreference.user_id == x_user_id)
        )
        user_pref = user_pref_result.scalars().first()

        if user_pref and user_pref.system_prompt:
            user_system_prompt_content = user_pref.system_prompt
            print(f"[DEBUG] 사용자 {x_user_id}의 맞춤 시스템 프롬프트 적용: '{user_system_prompt_content}'")
        else:
            if _default_system_prompt_cache:
                user_system_prompt_content = _default_system_prompt_cache
                print(f"[DEBUG] 사용자 {x_user_id}에게 캐시된 기본 시스템 프롬프트 적용: '{user_system_prompt_content}'")
            else:
                default_prompt_result = await session.execute(
                    select(UserPreference).where(UserPreference.user_id == "default_system").order_by(UserPreference.id)
                )
                default_prompt_db = default_prompt_result.scalars().first()
                if default_prompt_db and default_prompt_db.system_prompt:
                    user_system_prompt_content = default_prompt_db.system_prompt
                    _default_system_prompt_cache = user_system_prompt_content
                    print(f"[DEBUG] 사용자 {x_user_id}에게 DB 기본 프롬프트 적용 및 캐시 저장: '{user_system_prompt_content}'")
                else:
                    user_system_prompt_content = "You are a friendly and helpful AI assistant."
                    _default_system_prompt_cache = user_system_prompt_content
                    print(f"[DEBUG] 사용자 {x_user_id}에게 최후의 기본 시스템 프롬프트 적용 및 캐시 저장: '{user_system_prompt_content}'")

        # 2. 이전 대화 기록 가져오기
        previous_chats_result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == x_user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(20)
        )
        db_previous_chats = previous_chats_result.scalars().all()
        db_previous_chats.reverse()

        for chat_entry in db_previous_chats:
            openai_chat_history_list.append({"role": "user", "content": chat_entry.user_message})
            if chat_entry.bot_response:
                openai_chat_history_list.append({"role": "assistant", "content": chat_entry.bot_response})

    openai_chat_history_list.append({"role": "user", "content": message})

    context_text_for_prompt = ""
    referenced_docs_for_response_display = []
    if index.ntotal > 0:
        retrieved_documents = await search_similar_documents(message, x_user_id)
        if retrieved_documents:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서 수: {len(retrieved_documents)}")
            if len(retrieved_documents) > 0:
                print(f"[DEBUG] 사용자 {x_user_id}의 첫 번째 검색된 문서 내용 (일부): {retrieved_documents[0][:100]}...")
            # OpenAI 프롬프트에는 \\n을 그대로 사용 (모델이 \\n을 보고 줄바꿈으로 인식하도록)
            context_text_for_prompt = "\\n".join(retrieved_documents) 
            referenced_docs_for_response_display = retrieved_documents
        else:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 관련 문서를 찾지 못했습니다.")
    else:
        print(f"[DEBUG] 인덱스에 문서가 없습니다. 사용자 {x_user_id}에 대한 검색을 건너뜁니다.")

    reference_document_section_content = ""
    if context_text_for_prompt:
        reference_document_section_content = f"""다음은 사용자가 업로드한 문서에서 현재 대화와 관련성이 높은 내용입니다. 이 내용을 최우선으로 참고하여 사용자의 질문에 답변해주세요.
만약 이 내용만으로 답변이 부족하다면, 그 사실을 명확히 밝히고 일반적인 지식을 활용할 수 있습니다.
답변 시, 어떤 문서 내용을 참고했는지 간략히 언급하면 좋습니다.

[참고 문서 내용 시작]
{context_text_for_prompt}
[참고 문서 내용 끝]
"""
    else:
        reference_document_section_content = "현재 사용자가 업로드한 문서 중 대화와 관련된 내용을 찾지 못했습니다. 일반적인 지식을 바탕으로 답변해주세요."

    final_system_prompt = BASE_SYSTEM_PROMPT_TEMPLATE.format(
        user_specific_prompt_or_default=user_system_prompt_content,
        reference_document_section=reference_document_section_content
    ).strip()
    
    print(f"[DEBUG] 최종 시스템 프롬프트 (사용자 {x_user_id}):\\n{final_system_prompt}")

    messages_to_send_to_openai = [{"role": "system", "content": final_system_prompt}] + openai_chat_history_list

    async def event_stream():
        full_response_content = ""
        try:
            openai_response_stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_to_send_to_openai,
                stream=True
            )
            for chunk in openai_response_stream:
                content_piece = getattr(chunk.choices[0].delta, "content", None)
                if content_piece:
                    full_response_content += content_piece
                    # 모델이 생성한 '\\\\n'을 실제 줄바꿈 문자 '\\n'으로 변경하여 클라이언트에 전달
                    # 수정 전: yield f"data: {content_piece.replace('\\\\n', '\\n')}\\n\\n"
                    processed_content_piece = content_piece.replace('\\\\n', '\\n')
                    yield f"data: {processed_content_piece}\\n\\n"
                    await asyncio.sleep(0)

            async with async_session() as session:
                new_exchange_record = ChatHistory(
                    user_id=x_user_id,
                    user_message=message,
                    bot_response=full_response_content, # 모델이 생성한 '\\n' 포함 원본 저장
                    created_at=func.now()
                )
                session.add(new_exchange_record)
                await session.commit()

            if referenced_docs_for_response_display:
                yield f"data: \\n\\n[참고한 문단]\\n\\n"
                for idx, doc_content_item in enumerate(referenced_docs_for_response_display, 1):
                    # 모델이 생성한 '\\n'을 실제 줄바꿈 문자 '\\n'으로 변경하여 클라이언트에 전달
                    processed_doc_content = doc_content_item.replace('\\\\n', '\\n') 
                    yield f"data: [문단 {idx}]\\n{processed_doc_content}\\n\\n"

            # yield f"data: \\n\\n[DONE]\\n\\n" # 기존 코드
            yield f"data: [DONE]\\n\\n" # 수정된 코드

        except Exception as e:
            import traceback
            error_message = f"오류 발생 in event_stream: {str(e)}"
            print(error_message)
            traceback.print_exc()
            yield f"data: [ERROR] {error_message}\\n\\n"

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