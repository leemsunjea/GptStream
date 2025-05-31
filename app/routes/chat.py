# app/routes/chat.py

import asyncio
import traceback
from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI
from sqlalchemy import func, select

from app.config import settings
from db.models import ChatHistory, UserPreference
from db.database import async_session
from app.vector_db import index, search_similar_documents, search_recent_documents_first

router = APIRouter()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# 간소화된 포맷팅 규칙
FORMATTING_RULE = "답변 시 문장 끝에 줄바꿈을 사용하여 가독성을 높여주세요."

# ========================================================================================
# 메시지 구성 함수들
# ========================================================================================

def build_system_message(user_role: str = None, reference_docs: str = None) -> str:
    """시스템 메시지를 구성합니다."""
    base_role = user_role or "당신은 친절하고 도움이 되는 AI 어시스턴트입니다."
    
    components = [f"역할: {base_role}"]
    
    if reference_docs:
        components.append(f"참조 문서:\n{reference_docs}")
    
    components.append(FORMATTING_RULE)
    
    return "\n\n".join(components)

def build_messages(system_msg: str, chat_history: list, current_message: str) -> list:
    """OpenAI API용 메시지 리스트를 구성합니다."""
    messages = [{"role": "system", "content": system_msg}]
    
    # 대화 기록 추가
    for chat in chat_history:
        messages.append({"role": "user", "content": chat.user_message})
        if chat.bot_response:
            messages.append({"role": "assistant", "content": chat.bot_response})
    
    # 현재 메시지 추가
    messages.append({"role": "user", "content": current_message})
    
    return messages

@router.post("/stream")
async def chat_stream(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    message = data.get("message", "")
    
    print(f"[INFO] 채팅 스트림 시작 - 사용자: {x_user_id}, 메시지: '{message[:50]}...'")

    # 1. 사용자 설정 및 대화 기록 조회
    user_system_prompt_content = ""
    chat_history = []
    
    async with async_session() as session:
        # 사용자별 시스템 프롬프트 가져오기
        user_pref_result = await session.execute(
            select(UserPreference).where(UserPreference.user_id == x_user_id)
        )
        user_pref = user_pref_result.scalars().first()

        if user_pref and user_pref.system_prompt:
            user_system_prompt_content = user_pref.system_prompt
            print(f"[DEBUG] 사용자 {x_user_id}의 맞춤 시스템 프롬프트 적용")
        else:
            user_system_prompt_content = "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 정확하고 유용한 답변을 제공해주세요."
            print(f"[DEBUG] 사용자 {x_user_id}에게 기본 시스템 프롬프트 적용")

        # 이전 대화 기록 가져오기 (최근 20개)
        previous_chats_result = await session.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == x_user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(20)
        )
        db_previous_chats = previous_chats_result.scalars().all()
        chat_history = list(reversed(db_previous_chats))  # 시간순 정렬

    # 2. 문서 검색 및 참조 처리
    context_text_for_prompt = ""
    referenced_docs_for_response_display = []
    recommended_response_style = ""
    
    # 간소화된 키워드 감지
    recent_keywords = ["방금", "최근", "업로드", "요약해", "정리해", "설명해"]
    toc_keywords = ["목차", "차례", "구성", "전체", "구조"]
    
    is_recent_doc_query = any(keyword in message for keyword in recent_keywords)
    is_toc_query = any(keyword in message for keyword in toc_keywords)
    
    if index.ntotal > 0:
        print(f"[DEBUG] FAISS 인덱스에 {index.ntotal}개 벡터 로드됨. 문서 검색 시작...")
        
        # 문서 검색
        if is_recent_doc_query:
            print(f"[DEBUG] 최근 업로드 문서 관련 질문 감지")
            retrieved_documents_details = await search_recent_documents_first(message, x_user_id)
        elif is_toc_query:
            print(f"[DEBUG] 목차 관련 질문 감지")
            retrieved_documents_details = await search_similar_documents(message, x_user_id)
        else:
            retrieved_documents_details = await search_similar_documents(message, x_user_id)
            
        if retrieved_documents_details:
            print(f"[DEBUG] 검색된 관련 문서 수: {len(retrieved_documents_details)}")
            
            # 컨텍스트 생성
            context_parts = []
            response_styles = []
            for doc_detail in retrieved_documents_details:
                text_content = doc_detail.get('text', '')
                title = doc_detail.get('title', '제목 없음')
                summary = doc_detail.get('summary', '')
                response_style = doc_detail.get('response_style', '')
                pdf_name = doc_detail.get('pdf_name', '')
                
                context_part = f"[문서: {pdf_name}]\n제목: {title}"
                if summary:
                    context_part += f"\n요약: {summary}"
                context_part += f"\n내용: {text_content}"
                context_parts.append(context_part)
                
                if response_style:
                    response_styles.append(response_style)
            
            context_text_for_prompt = "\n\n".join(context_parts)
            
            if response_styles:
                recommended_response_style = max(set(response_styles), key=response_styles.count)
                print(f"[DEBUG] 권장 응답 스타일: {recommended_response_style}")
            
            referenced_docs_for_response_display = retrieved_documents_details
        else:
            print(f"[DEBUG] 검색된 관련 문서가 없습니다.")
    else:
        print(f"[DEBUG] FAISS 인덱스가 비어있습니다.")

    # 3. 참조 문서 내용 구성
    reference_document_content = ""
    if context_text_for_prompt:
        style_instruction = ""
        if recommended_response_style:
            style_instruction = f"\n\n응답 스타일 지침: {recommended_response_style} 스타일로 답변해주세요."
        
        toc_instruction = ""
        if is_toc_query:
            toc_instruction = f"\n\n특별 지침: 사용자가 목차, 차례, 구성에 대해 질문했습니다. 문서의 전체 구조와 목차 정보를 최대한 상세하게 정리하여 제공해주세요."
        
        reference_document_content = f"""다음은 사용자가 업로드한 문서에서 관련성이 높은 내용입니다. 이 내용을 최우선으로 참고하여 답변해주세요.{style_instruction}{toc_instruction}

[참고 문서 내용]
{context_text_for_prompt}
[참고 문서 내용 끝]
"""
    else:
        reference_document_content = "현재 업로드된 문서 중 관련된 내용을 찾지 못했습니다. 일반적인 지식을 바탕으로 답변해주세요."

    # 4. 시스템 메시지 구성
    system_message = build_system_message(
        user_role=user_system_prompt_content,
        reference_docs=reference_document_content
    )

    # 5. 최종 OpenAI 메시지 구조 구성
    messages_to_send_to_openai = build_messages(system_message, chat_history, message)

    print(f"[DEBUG] OpenAI 메시지 구조 구성 완료 - 총 {len(messages_to_send_to_openai)}개 메시지")
    print(f"[DEBUG] 시스템 메시지 길이: {len(system_message)} 문자")

    # 6. OpenAI API 스트리밍 실행
    async def event_stream():
        full_response_content = ""
        buffer = ""
        buffer_size_limit = 15
        
        try:
            print(f"[DEBUG] OpenAI API 호출 시작 - 모델: gpt-3.5-turbo")
            openai_response_stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_to_send_to_openai,
                stream=True,
                temperature=0.7
            )
            
            for chunk in openai_response_stream:
                content_piece = getattr(chunk.choices[0].delta, "content", None)
                if content_piece:
                    full_response_content += content_piece
                    buffer += content_piece
                    
                    should_flush = (
                        len(buffer) >= buffer_size_limit or
                        '\n' in buffer
                    )
                    
                    if should_flush and buffer.strip():
                        yield f"data: {buffer}\n"
                        yield "\n"
                        buffer = ""
                        await asyncio.sleep(0.01)
            
            # 남은 버퍼 전송
            if buffer.strip():
                yield f"data: {buffer}\n"
                yield "\n"
            
            # 완료 신호
            yield f"data: [DONE]\n"
            yield "\n"
            
            # 대화 기록 저장
            async with async_session() as session:
                new_chat = ChatHistory(
                    user_id=x_user_id,
                    user_message=message,
                    bot_response=full_response_content
                )
                session.add(new_chat)
                await session.commit()
                print(f"[DEBUG] 대화 기록 저장 완료")

        except Exception as e:
            print(f"[ERROR] 스트리밍 중 오류 발생: {str(e)}")
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
async def add_user_prompt(request: Request, x_user_id: str = Header(..., description="클라이언트 UUID")):
    data = await request.json()
    new_prompt_text = data.get("prompt", "")

    if not new_prompt_text:
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
            user_pref.updated_at = func.now()
            print(f"[DEBUG] 사용자 {x_user_id}의 프롬프트 업데이트: '{previous_prompt}' -> '{new_prompt_text}'")
        else:
            # 새 프롬프트 생성
            user_pref = UserPreference(user_id=x_user_id, system_prompt=new_prompt_text)
            session.add(user_pref)
            print(f"[DEBUG] 사용자 {x_user_id}의 새 프롬프트 생성: '{new_prompt_text}'")
        
        await session.commit()
        await session.refresh(user_pref)

    return {"success": True, "new_system_prompt": user_pref.system_prompt}