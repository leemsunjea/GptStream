# app/routes/chat.py

import asyncio
from fastapi import APIRouter, Request, Header, HTTPException # HTTPException 추가
from fastapi.responses import StreamingResponse, JSONResponse # JSONResponse 추가
import openai
from app.config import settings
from db.models import ChatHistory, UserPreference # UserPreference 임포트 추가
from db.database import async_session
from app.vector_db import get_embedding_async as get_embedding, index, doc_store, search_similar_documents, search_recent_documents_first # search_recent_documents_first 추가
from openai import OpenAI
from sqlalchemy import func, select, delete # delete 추가
from db.models import UserPreference, ChatHistory # ChatHistory 임포트 추가
import traceback
from sqlalchemy import func, select, delete # delete 추가
from db.models import UserPreference, ChatHistory # ChatHistory 임포트 추가
import traceback

from app.vector_db import task_statuses

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Global cache for default system prompt
_default_system_prompt_cache = None

# ========================================================================================
# 줄바꿈 포맷팅 규칙 - 시스템 프롬프트용 상수
# ========================================================================================

FORMATTING_RULES_SYSTEM_PROMPT = """=== 필수 응답 형식 규칙 ===

**중요: 다음 줄바꿈 규칙을 반드시 준수하세요**

1. 문장이 끝날 때마다 \\n을 추가하세요
2. 새로운 단락 시작 시 \\n\\n을 사용하세요
3. 목록 항목 끝에 \\n을 추가하세요
4. 제목/헤더 뒤에 \\n\\n을 추가하세요
5. 긴 텍스트를 줄바꿈 없이 연속 작성하지 마세요

**올바른 예시:**
"안녕하세요.\\n도움이 필요하시군요.\\n\\n저는 AI 어시스턴트입니다.\\n무엇을 도와드릴까요?"

**잘못된 예시:**
"안녕하세요. 도움이 필요하시군요. 저는 AI 어시스턴트입니다. 무엇을 도와드릴까요?"

이 규칙을 지키지 않으면 텍스트가 읽기 어려운 형태로 표시됩니다."""

# ========================================================================================
# OpenAI 메시지 구조 구성 함수들 - 완전히 새로운 설계
# ========================================================================================

def build_system_message(user_role: str = None, reference_docs: str = None) -> str:
    """시스템 메시지를 구성합니다."""
    
    # 기본 역할 설정
    base_role = user_role or "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 정확하고 유용한 답변을 제공해주세요."
    
    # 시스템 메시지 구성 요소들
    system_components = []
    
    # 1. 기본 역할 정의
    system_components.append(f"=== AI 어시스턴트 역할 ===\n{base_role}")
    
    # 2. 참조 문서 (있는 경우)
    if reference_docs:
        system_components.append(f"=== 참조 문서 내용 ===\n{reference_docs}")
    
    # 3. 필수 응답 형식 규칙 (항상 포함) - 별도 상수 사용
    system_components.append(FORMATTING_RULES_SYSTEM_PROMPT)
    
    return "\n\n".join(system_components)

def build_conversation_messages(chat_history: list) -> list:
    """대화 기록을 OpenAI 메시지 형식으로 변환합니다."""
    messages = []
    
    for chat_entry in chat_history:
        # 사용자 메시지 추가
        messages.append({
            "role": "user", 
            "content": chat_entry.user_message
        })
        
        # 어시스턴트 응답 추가 (있는 경우)
        if chat_entry.bot_response:
            messages.append({
                "role": "assistant", 
                "content": chat_entry.bot_response
            })
    
    return messages

def build_openai_message_structure(system_message: str, conversation_history: list, current_user_message: str) -> list:
    """
    최종 OpenAI API 메시지 구조를 구성합니다.
    
    구조:
    [
        {"role": "system", "content": "시스템 지시 프롬프트"},
        {"role": "user", "content": "이전 사용자 입력 1"},
        {"role": "assistant", "content": "이전 GPT 응답 1 (선택 사항)"},
        {"role": "user", "content": "이전 사용자 입력 2"},
        {"role": "assistant", "content": "이전 GPT 응답 2 (선택 사항)"},
        ...
        {"role": "user", "content": "현재 사용자 입력"}
    ]
    """
    messages = []
    
    # 1. 시스템 메시지 (항상 첫 번째)
    messages.append({
        "role": "system",
        "content": system_message
    })
    
    # 2. 이전 대화 기록 추가 (시간순)
    messages.extend(conversation_history)
    
    # 3. 현재 사용자 메시지 추가 (마지막)
    messages.append({
        "role": "user",
        "content": current_user_message
    })
    
    return messages

def validate_message_structure(messages: list) -> bool:
    """메시지 구조가 올바른지 검증합니다."""
    if not messages:
        return False
    
    # 첫 번째 메시지는 반드시 system이어야 함
    if messages[0].get("role") != "system":
        return False
    
    # 마지막 메시지는 반드시 user여야 함
    if messages[-1].get("role") != "user":
        return False
    
    # 모든 메시지에 role과 content가 있는지 확인
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ["system", "user", "assistant"]:
            return False
        if not isinstance(msg["content"], str) or not msg["content"].strip():
            return False
    
    return True

# 기존 템플릿 제거하고 새로운 구조 사용
# BASE_SYSTEM_PROMPT_TEMPLATE 제거

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
    
    # 키워드 감지
    recent_doc_keywords = [
        "방금 업로드", "최근 업로드", "업로드한 문서", "내가 올린", "방금 올린", "최근에 올린",
        "방금 등록", "최근 등록", "등록한 문서", "방금 추가", "최근 추가", "추가한 문서",
        "방금 저장", "최근 저장", "저장한 문서", "새로 올린", "새로 업로드", "새 문서",
        "요약해", "정리해", "설명해", "알려줘", "뭐가 있어", "어떤 내용"
    ]
    
    table_of_contents_keywords = [
        "목차", "차례", "목록", "구성", "내용", "인덱스", "개요", "구조",
        "table of contents", "contents", "index", "outline", "structure",
        "전체", "모든", "다 알려", "전부", "리스트", "항목"
    ]
    
    is_recent_doc_query = any(keyword in message for keyword in recent_doc_keywords)
    is_toc_query = any(keyword in message for keyword in table_of_contents_keywords)
    
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

    # 5. 대화 기록을 OpenAI 메시지 형식으로 변환
    conversation_messages = build_conversation_messages(chat_history)

    # 6. 최종 OpenAI 메시지 구조 구성
    messages_to_send_to_openai = build_openai_message_structure(
        system_message=system_message,
        conversation_history=conversation_messages,
        current_user_message=message
    )

    # 7. 메시지 구조 검증
    if not validate_message_structure(messages_to_send_to_openai):
        print(f"[ERROR] 메시지 구조 검증 실패")
        raise HTTPException(status_code=500, detail="메시지 구조 구성 오류")

    print(f"[DEBUG] OpenAI 메시지 구조 구성 완료 - 총 {len(messages_to_send_to_openai)}개 메시지")
    print(f"[DEBUG] 시스템 메시지 길이: {len(system_message)} 문자")
    print(f"[DEBUG] 대화 기록: {len(conversation_messages)}개 메시지")

    # 8. OpenAI API 스트리밍 실행
    async def event_stream():
        full_response_content = ""
        buffer = ""
        buffer_size_limit = 15
        
        try:
            print(f"[DEBUG] OpenAI API 호출 시작 - 모델: gpt-3.5-turbo")
            openai_response_stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages_to_send_to_openai+'=== 필수 응답 형식 규칙 ===\\n\\n**중요: 다음 줄바꿈 규칙을 반드시 준수하세요**\\n\\n1. 문장이 끝날 때마다 \\\\n을 추가하세요\\n2. 새로운 단락 시작 시 \\\\n\\\\n을 사용하세요\\n3. 목록 항목 끝에 \\\\n을 추가하세요\\n4. 제목/헤더 뒤에 \\\\n\\\\n을 추가하세요\\n5. 긴 텍스트를 줄바꿈 없이 연속 작성하지 마세요\\n\\n**올바른 예시:**\\n\"안녕하세요.\\\\n도움이 필요하시군요.\\\\n\\\\n저는 AI 어시스턴트입니다.\\\\n무엇을 도와드릴까요?\"\\n\\n**잘못된 예시:**\\n\"안녕하세요. 도움이 필요하시군요. 저는 AI 어시스턴트입니다. 무엇을 도와드릴까요?\"\\n\\n이 규칙을 지키지 않으면 텍스트가 읽기 어려운 형태로 표시됩니다.',
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