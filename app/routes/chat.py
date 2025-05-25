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

이제 사용자의 다음 질문에 답변해주세요. 답변은 한국어로 작성하고, 답변 내용에 줄바꿈이 필요하면 '\\n'을 사용하세요.
"""

@router.post("/chat/stream")
async def chat_stream(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    message = data.get("message", "")

    openai_chat_history_list = []
    user_system_prompt_content = ""

    async with async_session() as session:
        # 1. 사용자별 시스템 프롬프트 가져오기
        user_pref_result = await session.execute(
            select(UserPreference).where(UserPreference.user_id == x_user_id)
        )
        user_pref = user_pref_result.scalars().first()

        if user_pref and user_pref.system_prompt:
            user_system_prompt_content = user_pref.system_prompt
            print(f"[DEBUG] 사용자 {x_user_id}의 맞춤 시스템 프롬프트 적용: '{user_system_prompt_content}'")
        else:
            # 기본 프롬프트 설정 (캐시 대신 직접 정의)
            user_system_prompt_content = "You are a friendly and helpful AI assistant."
            print(f"[DEBUG] 사용자 {x_user_id}에게 기본 시스템 프롬프트 적용: '{user_system_prompt_content}'")

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

    # 문서 검색 및 참조 처리 (기존 로직 유지)
    context_text_for_prompt = ""
    referenced_docs_for_response_display = []
    if index.ntotal > 0:
        retrieved_documents = await search_similar_documents(message, x_user_id)
        if retrieved_documents:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서 수: {len(retrieved_documents)}")
            context_text_for_prompt = "\\n".join(retrieved_documents)
            referenced_docs_for_response_display = retrieved_documents

    reference_document_section_content = ""
    if context_text_for_prompt:
        reference_document_section_content = f"""다음은 사용자가 업로드한 문서에서 현재 대화와 관련성이 높은 내용입니다. 이 내용을 최우선으로 참고하여 사용자의 질문에 답변해주세요.
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
                    processed_content_piece = content_piece.replace('\\n', '\n')
                    lines = processed_content_piece.split('\n')
                    for line in lines:
                        if line:
                            yield f"data: {line}\n"
                    yield "\n"
                    await asyncio.sleep(0)

            if referenced_docs_for_response_display:
                yield f"data: [참고한 문단]\n"
                for idx, doc_content_item in enumerate(referenced_docs_for_response_display, 1):
                    processed_doc_content = doc_content_item.replace('\\n', '\n')
                    lines = processed_doc_content.split('\n')
                    for line in lines:
                        if line:
                            yield f"data: [문단 {idx}] {line}\n"
                yield "\n"

            yield f"data: [DONE]\n"
            yield "\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n"
            yield "\n"

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