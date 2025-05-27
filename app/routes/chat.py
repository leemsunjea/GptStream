# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request, Header, HTTPException # HTTPException 추가
from fastapi.responses import StreamingResponse, JSONResponse # JSONResponse 추가
import openai
import numpy as np
from app.config import settings
from db.models import ChatHistory, UserPreference # UserPreference 임포트 추가
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store, search_similar_documents # search_similar_documents 추가
from openai import OpenAI
from sqlalchemy import func, select, delete # delete 추가
from db.models import UserPreference, ChatHistory # ChatHistory 임포트 추가
import traceback

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Global cache for default system prompt
_default_system_prompt_cache = None

# 새로운 기본 시스템 프롬프트 템플릿
BASE_SYSTEM_PROMPT_TEMPLATE = """
{user_specific_prompt_or_default}

{reference_document_section}

이제 사용자의 다음 질문에 답변해주세요. 답변은 한국어로 작성하고, 답변 내용에 줄바꿈이 필요하면 '\\\\n'을 사용하세요.
"""

@router.post("/stream") # 경로 수정: "/chat/stream" -> "/stream"
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

    # 문서 검색 및 참조 처리
    context_text_for_prompt = ""
    referenced_docs_for_response_display = [] # 전체 문서 메타데이터를 담을 리스트
    # 검색된 각 문서는 이제 텍스트뿐만 아니라 title, summary, response_style, created_at, pdf_name 등의 메타데이터를 포함한 딕셔너리입니다.
    if index.ntotal > 0:
        retrieved_documents_details = await search_similar_documents(message, x_user_id)
        if retrieved_documents_details:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서 수: {len(retrieved_documents_details)}")
            
            # 프롬프트에 포함할 컨텍스트 생성 (예: 각 문서의 텍스트와 일부 메타데이터)
            context_parts = []
            for doc_detail in retrieved_documents_details:
                # doc_detail은 이제 딕셔너리입니다.
                text_content = doc_detail.get('text', '')
                title = doc_detail.get('title', '제목 없음')
                # 필요에 따라 summary, created_at 등 다른 메타데이터도 여기에 추가할 수 있습니다.
                context_parts.append(f"문서 제목: {title}\n내용: {text_content}") 
            context_text_for_prompt = "\n\n".join(context_parts)
            
            # 응답에 표시할 참조 정보 (예: 전체 메타데이터 또는 선택적 정보)
            # 여기서는 검색된 각 항목(문단과 그 메타데이터)을 그대로 사용합니다.
            referenced_docs_for_response_display = retrieved_documents_details 

    reference_document_section_content = ""
    if context_text_for_prompt:
        reference_document_section_content = f"""다음은 사용자가 업로드한 문서에서 현재 대화와 관련성이 높은 내용입니다. 각 문서는 제목과 내용으로 구성되어 있습니다. 이 내용을 최우선으로 참고하여 사용자의 질문에 답변해주세요.
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
                    processed_content_piece = content_piece.replace('\\\\n', '\\n')
                    lines = processed_content_piece.split('\\n')
                    for line in lines:
                        if line:
                            yield f"data: {line}\\n"
                    yield "\\n"
                    await asyncio.sleep(0)

            # if referenced_docs_for_response_display:
            #     yield f"data: [참고한 문서 정보]\\n"
            #     for idx, doc_detail in enumerate(referenced_docs_for_response_display, 1):
            #         # doc_detail은 이제 딕셔너리입니다.
            #         title = doc_detail.get('title', '제목 없음')
            #         summary = doc_detail.get('summary', '요약 없음')
            #         pdf_name = doc_detail.get('pdf_name', '알 수 없는 PDF')
            #         created_at_iso = doc_detail.get('created_at', '시간 정보 없음')
            #         text_preview = doc_detail.get('text', '')[:100] # 문단 내용 미리보기
                    
            #         # 클라이언트에 전달할 정보 구성 (예시)
            #         # 상세 정보나 문단 전체 텍스트를 보낼 수도 있습니다.
            #         # 여기서는 제목, 요약, 파일명, 생성 시간, 문단 미리보기를 전달합니다.
            #         doc_info_line = f"[문서 {idx}] 제목: {title} (파일명: {pdf_name}, 업로드: {created_at_iso})\\n요약: {summary}\\n문단 미리보기: {text_preview}..."
                    
            #         # 여러 줄로 된 정보를 안전하게 전송하기 위해 각 줄을 data: 로 시작하도록 처리
            #         for line in doc_info_line.split('\\n'):
            #             if line:
            #                 yield f"data: {line}\\n"
            #         yield f"data: ---\\n" # 문서 정보 구분자
            #     yield "\\n"

            yield f"data: [DONE]\\n"
            yield "\\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n"
            yield "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.post("/reset_prompt")
async def reset_user_prompt(x_user_id: str = Header(..., description="클라이언트 UUID")):
    """특정 사용자의 시스템 프롬프트를 기본값으로 초기화합니다."""
    logs = []
    logs.append(f"사용자 [{x_user_id}] 프롬프트 초기화 시작...")
    print(f"[INFO] 사용자 {x_user_id}의 프롬프트 초기화 요청 수신.")

    async with async_session() as session:
        async with session.begin(): # 트랜잭션 시작
            try:
                # 사용자 프롬프트 조회
                user_pref_result = await session.execute(
                    select(UserPreference).where(UserPreference.user_id == x_user_id)
                )
                user_pref = user_pref_result.scalars().first()

                if user_pref:
                    # 기본 프롬프트로 설정 (첫 번째 DEFAULT_SYSTEM_PROMPTS 또는 빈 문자열)
                    # main.py 에서 DEFAULT_SYSTEM_PROMPTS 를 가져올 수 없으므로, 여기서 직접 정의하거나 다른 방식으로 관리 필요
                    # 여기서는 간단하게 빈 문자열로 초기화하거나, 특정 기본값을 설정합니다.
                    default_prompt_for_user = "You are a friendly and helpful AI assistant." # 예시 기본 프롬프트
                    user_pref.system_prompt = default_prompt_for_user
                    user_pref.updated_at = func.now() # 업데이트 시간 기록
                    await session.commit() # 변경사항 커밋
                    logs.append(f"사용자 [{x_user_id}]의 프롬프트를 기본값으로 성공적으로 초기화했습니다.")
                    print(f"[INFO] 사용자 {x_user_id}의 프롬프트 초기화 완료.")
                    return JSONResponse({
                        "success": True, 
                        "message": f"사용자 [{x_user_id}]의 프롬프트가 기본값으로 초기화되었습니다.",
                        "new_system_prompt": default_prompt_for_user,
                        "logs": logs
                    })
                else:
                    logs.append(f"사용자 [{x_user_id}]에 대한 프롬프트 설정을 찾을 수 없습니다. 초기화할 프롬프트가 없습니다.")
                    print(f"[INFO] 사용자 {x_user_id}의 프롬프트 설정 없음.")
                    # 이 경우 클라이언트에게 성공으로 응답할지, 오류로 응답할지 결정 필요
                    return JSONResponse({
                        "success": True, # 또는 False, 상황에 따라
                        "message": f"사용자 [{x_user_id}]에 대한 프롬프트 설정을 찾을 수 없습니다.",
                        "logs": logs
                    }, status_code=200) # 또는 404

            except Exception as e:
                await session.rollback() # 오류 발생 시 롤백
                error_tb = traceback.format_exc()
                logs.append(f"프롬프트 초기화 중 서버 오류 발생: {str(e)}")
                print(f"[ERROR] 사용자 {x_user_id} 프롬프트 초기화 중 오류: {e}\n{error_tb}")
                return JSONResponse(
                    status_code=500, 
                    content={"success": False, "detail": f"프롬프트 초기화 중 서버 오류 발생: {str(e)}", "logs": logs}
                )

@router.post("/add_prompt")
async def add_user_prompt(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")): # x_user_id 추가
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