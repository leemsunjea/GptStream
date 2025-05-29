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

# 새로운 기본 시스템 프롬프트 템플릿
BASE_SYSTEM_PROMPT_TEMPLATE = """
{user_specific_prompt_or_default}

{reference_document_section}

**중요: 반드시 마크다운 형식으로 답변해주세요!**

답변은 한국어로 작성하되, 다음 마크다운 형식을 **반드시** 활용하여 구조화된 답변을 제공해주세요:

**필수 마크다운 형식 규칙:**
- **제목 사용**: # (대제목), ## (중제목), ### (소제목)을 적극 활용하세요
- **강조 표시**: 중요한 내용은 **굵게** 또는 *기울임*으로 강조하세요
- **목록 활용**: 
  - 순서가 있는 내용: 1. 2. 3. (번호 목록)
  - 순서가 없는 내용: - 또는 * (점 목록)
- **코드 표시**: 
  - 짧은 코드: `인라인 코드`
  - 긴 코드: ```코드 블록```
- **줄바꿈**: 단락 구분을 위해 `\n`을 사용하세요 (두 번 사용하면 빈 줄)

**답변 구조 예시:**
```
# 답변 제목

## 주요 내용
설명 내용...

### 세부 사항
1. 첫 번째 항목
2. 두 번째 항목

**중요한 점**: 강조할 내용

- 추가 정보 1
- 추가 정보 2

## 결론
마무리 내용...
```

위 형식을 참고하여 **반드시 마크다운과 적절한 줄바꿈을 사용해** 구조화된 답변을 제공해주세요.
각 섹션 사이에는 빈 줄을 추가하고, 목록이나 단락을 구분할 때는 줄바꿈을 활용하여 가독성을 높여주세요.
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
    recommended_response_style = ""
    
    # "방금 업로드한 문서", "최근 업로드한", "업로드한 문서" 등의 키워드 감지
    recent_doc_keywords = [
        "방금 업로드", "최근 업로드", "업로드한 문서", "내가 올린", "방금 올린", "최근에 올린",
        "방금 등록", "최근 등록", "등록한 문서", "방금 추가", "최근 추가", "추가한 문서",
        "방금 저장", "최근 저장", "저장한 문서", "새로 올린", "새로 업로드", "새 문서",
        "요약해", "정리해", "설명해", "알려줘", "뭐가 있어", "어떤 내용"
    ]
    
    # 목차 관련 키워드 확장
    table_of_contents_keywords = [
        "목차", "차례", "목록", "구성", "내용", "인덱스", "개요", "구조",
        "table of contents", "contents", "index", "outline", "structure",
        "전체", "모든", "다 알려", "전부", "리스트", "항목"
    ]
    
    is_recent_doc_query = any(keyword in message for keyword in recent_doc_keywords)
    is_toc_query = any(keyword in message for keyword in table_of_contents_keywords)
    
    if index.ntotal > 0:
        print(f"[DEBUG] FAISS 인덱스에 {index.ntotal}개 벡터 로드됨. 문서 검색 시작...")
        
        # 최근 문서 쿼리인 경우 특별 처리
        if is_recent_doc_query:
            print(f"[DEBUG] 최근 업로드 문서 관련 질문 감지: '{message}'")
            # 최근 업로드된 문서를 우선적으로 검색
            retrieved_documents_details = await search_recent_documents_first(message, x_user_id)
        elif is_toc_query:
            print(f"[DEBUG] 목차 관련 질문 감지: '{message}'")
            # 목차 관련 검색 수행 (확장된 검색)
            retrieved_documents_details = await search_similar_documents(message, x_user_id)
        else:
            retrieved_documents_details = await search_similar_documents(message, x_user_id)
            
        if retrieved_documents_details:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서 수: {len(retrieved_documents_details)}")
            
            # 프롬프트에 포함할 컨텍스트 생성 (메타데이터 포함)
            context_parts = []
            response_styles = []
            for doc_detail in retrieved_documents_details:
                text_content = doc_detail.get('text', '')
                title = doc_detail.get('title', '제목 없음')
                summary = doc_detail.get('summary', '')
                response_style = doc_detail.get('response_style', '')
                pdf_name = doc_detail.get('pdf_name', '')
                
                # 메타데이터가 포함된 컨텍스트 생성
                context_part = f"[문서: {pdf_name}]\n제목: {title}"
                if summary:
                    context_part += f"\n요약: {summary}"
                context_part += f"\n내용: {text_content}"
                context_parts.append(context_part)
                
                # 응답 스타일 수집
                if response_style:
                    response_styles.append(response_style)
            
            context_text_for_prompt = "\n\n".join(context_parts)
            print(f"[DEBUG] 생성된 컨텍스트 길이: {len(context_text_for_prompt)} 문자")
            
            # 가장 빈번한 응답 스타일 선택
            if response_styles:
                recommended_response_style = max(set(response_styles), key=response_styles.count)
                print(f"[DEBUG] 권장 응답 스타일: {recommended_response_style}")
            
            referenced_docs_for_response_display = retrieved_documents_details
        else:
            print(f"[DEBUG] 사용자 {x_user_id}에 대해 검색된 관련 문서가 없습니다.")
    else:
        print(f"[DEBUG] FAISS 인덱스가 비어있습니다 (ntotal: {index.ntotal}). 문서 검색을 건너뜁니다.") 

    reference_document_section_content = ""
    if context_text_for_prompt:
        style_instruction = ""
        if recommended_response_style:
            style_instruction = f"\n\n응답 스타일 지침: {recommended_response_style} 스타일로 답변해주세요."
        
        # 목차 관련 질문인 경우 특별한 지침 추가
        toc_instruction = ""
        if is_toc_query:
            toc_instruction = f"\n\n특별 지침: 사용자가 목차, 차례, 구성에 대해 질문했습니다. 문서의 전체 구조와 목차 정보를 최대한 상세하게 정리하여 제공해주세요. 각 장이나 섹션의 제목과 주요 내용을 포함하여 답변해주세요."
        
        reference_document_section_content = f"""다음은 사용자가 업로드한 문서에서 현재 대화와 관련성이 높은 내용입니다. 각 문서는 제목, 요약, 내용으로 구성되어 있습니다. 이 내용을 최우선으로 참고하여 사용자의 질문에 답변해주세요.{style_instruction}{toc_instruction}

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
        buffer = ""  # 토큰 버퍼
        buffer_size_limit = 15  # 버퍼 크기 축소 (더 빠른 업데이트)
        
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
                    buffer += content_piece
                    
                    # 마크다운 포맷 보호를 위한 전송 조건
                    should_flush = (
                        len(buffer) >= buffer_size_limit or
                        # 문장 단위 구분 (마크다운 안전)
                        buffer.endswith('. ') or 
                        buffer.endswith('.\n') or 
                        buffer.endswith('! ') or 
                        buffer.endswith('?\n') or
                        # 자연스러운 줄바꿈 지점
                        buffer.endswith('\n\n') or
                        # 리스트나 제목 등 마크다운 구조 완성
                        (buffer.count('\n') > 0 and (
                            buffer.strip().endswith(':') or  # 제목이나 리스트 시작
                            buffer.strip().startswith('#') or  # 헤딩 
                            buffer.strip().startswith('- ') or  # 리스트 항목
                            buffer.strip().startswith('* ') or  # 리스트 항목
                            buffer.strip().startswith('1. ') or  # 번호 리스트
                            buffer.strip().endswith('.')  # 문장 완성
                        ))
                    )
                    
                    if should_flush and buffer.strip():
                        # 원본 텍스트 그대로 전송 (클라이언트에서 마크다운 처리)
                        yield f"data: {buffer}\n"
                        yield "\n"  # SSE 메시지 구분
                        buffer = ""  # 버퍼 초기화
                        await asyncio.sleep(0.01)  # 지연 시간 단축 (10ms로 줄임)
            
            # 스트림 종료 후 남은 버퍼 전송
            if buffer.strip():
                yield f"data: {buffer}\n"
                yield "\n"
            
            # 스트림 완료 신호
            yield f"data: [DONE]\n"
            yield "\n"
            
            # 대화 기록을 DB에 저장
            async with async_session() as session:
                new_chat = ChatHistory(
                    user_id=x_user_id,
                    user_message=message,
                    bot_response=full_response_content
                )
                session.add(new_chat)
                await session.commit()
                print(f"[DEBUG] 대화 기록 저장 완료 - 사용자: {x_user_id}")

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