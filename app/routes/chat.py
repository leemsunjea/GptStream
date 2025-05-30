# app/routes/chat.py

import asyncio
import re
import html
from typing import Dict
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
# HTML 처리 클래스 - 독립 시스템에서 통합
# ========================================================================================

class AdvancedHTMLProcessor:
    """향상된 HTML 변환 및 파싱 프로세서"""
    
    def __init__(self):
        pass
    
    def convert_to_html(self, text_response: str) -> str:
        """텍스트를 HTML로 변환"""
        
        if not text_response:
            return '<div class="chat-response"><p>텍스트가 비어있습니다.</p></div>'
        
        # 1. HTML 특수문자 이스케이프
        html_escaped = html.escape(text_response)
        
        # 2. 헤더 처리 (### Header -> <h3>Header</h3>)
        html_with_headers = self._convert_headers_to_html(html_escaped)
        
        # 3. 목록 처리
        html_with_lists = self._convert_lists_to_html(html_with_headers)
        
        # 4. 강조 처리
        html_with_emphasis = self._convert_emphasis_to_html(html_with_lists)
        
        # 5. 줄바꿈 처리 (개선된)
        html_with_breaks = self._convert_linebreaks_to_html(html_with_emphasis)
        
        # 6. 최종 래핑
        final_html = f'<div class="chat-response">{html_with_breaks}</div>'
        
        return final_html
    
    def _convert_headers_to_html(self, text: str) -> str:
        """헤더를 HTML로 변환"""
        # ### Header -> <h3>Header</h3>
        text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        return text
    
    def _convert_lists_to_html(self, text: str) -> str:
        """목록을 HTML로 변환"""
        # - 또는 * 로 시작하는 목록 패턴
        list_pattern = r'(?:^|\n)((?:[-*]\s+.+(?:\n|$))+)'
        
        def replace_list(match):
            list_content = match.group(1).strip()
            items = re.findall(r'[-*]\s+(.+)', list_content)
            
            if items:
                html_items = [f'<li>{item.strip()}</li>' for item in items]
                return f'\n<ul>{"".join(html_items)}</ul>\n'
            
            return match.group(0)
        
        return re.sub(list_pattern, replace_list, text, flags=re.MULTILINE)
    
    def _convert_emphasis_to_html(self, text: str) -> str:
        """강조를 HTML로 변환"""
        # **bold** -> <strong>bold</strong>
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # *italic* -> <em>italic</em>
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        return text
    
    def _convert_linebreaks_to_html(self, text: str) -> str:
        """줄바꿈을 HTML로 변환 (개선된 방식)"""
        
        # 1. 연속된 줄바꿈을 단락으로 분리
        text = re.sub(r'\n\n+', '</p><p>', text)
        
        # 2. 단일 줄바꿈은 <br>로 변환
        text = text.replace('\n', '<br>')
        
        # 3. 전체를 <p> 태그로 감싸기 (헤더나 목록이 아닌 경우)
        if not text.startswith(('<h', '<ul', '<ol')):
            text = f'<p>{text}</p>'
        
        # 4. 빈 <p> 태그 제거
        text = re.sub(r'<p>\s*</p>', '', text)
        
        return text
    
    def validate_html_structure(self, html_content: str) -> Dict:
        """HTML 구조 검증"""
        
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "tag_count": {},
            "total_tags": 0,
            "balanced": True
        }
        
        # 태그 검증
        common_tags = ['div', 'p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'strong', 'em']
        
        for tag in common_tags:
            open_pattern = re.compile(f'<{tag}[^>]*>', re.IGNORECASE)
            close_pattern = re.compile(f'</{tag}>', re.IGNORECASE)
            
            open_count = len(open_pattern.findall(html_content))
            close_count = len(close_pattern.findall(html_content))
            
            validation["tag_count"][tag] = {"open": open_count, "close": close_count}
            validation["total_tags"] += open_count + close_count
            
            if open_count != close_count:
                validation["balanced"] = False
                validation["errors"].append(f"{tag} 태그가 균형이 맞지 않음 (열림: {open_count}, 닫힘: {close_count})")
        
        # br 태그 카운트 (self-closing)
        br_count = len(re.findall(r'<br[^>]*>', html_content))
        validation["tag_count"]["br"] = {"open": br_count, "close": 0}
        validation["total_tags"] += br_count
        
        # 기본 구조 검증
        if not html_content.strip():
            validation["valid"] = False
            validation["errors"].append("HTML 내용이 비어있음")
        
        if '<div class="chat-response">' not in html_content:
            validation["warnings"].append("chat-response 클래스가 없음")
        
        return validation
    
    def extract_text_from_html(self, html_content: str) -> str:
        """HTML에서 순수 텍스트 추출"""
        
        # 1. HTML 태그 제거
        text_only = re.sub(r'<[^>]+>', '', html_content)
        
        # 2. HTML 엔티티 디코딩
        text_decoded = html.unescape(text_only)
        
        # 3. 연속된 공백 정리
        text_clean = re.sub(r'\s+', ' ', text_decoded)
        
        return text_clean.strip()

# HTML 프로세서 인스턴스 생성
html_processor = AdvancedHTMLProcessor()

# ========================================================================================
# 줄바꿈 포맷팅 규칙 - 시스템 프롬프트용 상수
# ========================================================================================

FORMATTING_RULES_SYSTEM_PROMPT = """=== 필수 응답 형식 규칙 ===

**CRITICAL: 다음 줄바꿈 규칙을 반드시 준수하세요**

1. **문장 종료 시**: 모든 문장이 끝날 때마다 반드시 \\n을 추가하세요
2. **단락 구분 시**: 새로운 주제나 단락 시작 시 \\n\\n을 사용하세요  
3. **목록 항목**: 각 목록 항목 끝에 \\n을 추가하세요
4. **헤더/제목**: 제목이나 헤더 뒤에 \\n\\n을 추가하세요
5. **긴 텍스트 금지**: 3문장 이상을 줄바꿈 없이 연속 작성하지 마세요

**올바른 예시:**
"안녕하세요.\\n도움이 필요하시군요.\\n\\n저는 AI 어시스턴트입니다.\\n무엇을 도와드릴까요?\\n\\n다음과 같은 기능을 제공합니다:\\n- 질문 답변\\n- 문서 요약\\n- 정보 검색\\n"

**잘못된 예시:**
"안녕하세요. 도움이 필요하시군요. 저는 AI 어시스턴트입니다. 무엇을 도와드릴까요? 다음과 같은 기능을 제공합니다: 질문 답변, 문서 요약, 정보 검색"

**중요**: 이 규칙을 지키지 않으면 사용자가 읽기 어려운 형태로 텍스트가 표시됩니다. 반드시 준수해주세요."""

# ========================================================================================
# 고급 HTML 처리 클래스 - 독립 시스템에서 가져온 기능
# ========================================================================================

class AdvancedHTMLProcessor:
    """향상된 HTML 변환 및 파싱 프로세서"""
    
    def __init__(self):
        pass
    
    def convert_to_html(self, text_response: str) -> str:
        """텍스트를 HTML로 변환"""
        
        if not text_response:
            return ""
        
        # 1. HTML 특수문자 이스케이프
        html_escaped = html.escape(text_response)
        
        # 2. 헤더 처리 (### Header -> <h3>Header</h3>)
        html_with_headers = self._convert_headers_to_html(html_escaped)
        
        # 3. 목록 처리
        html_with_lists = self._convert_lists_to_html(html_with_headers)
        
        # 4. 강조 처리 (**bold**, *italic*)
        html_with_emphasis = self._convert_emphasis_to_html(html_with_lists)
        
        # 5. 줄바꿈 처리
        html_with_linebreaks = self._convert_linebreaks_to_html(html_with_emphasis)
        
        # 6. 최종 HTML 구조로 래핑
        final_html = f'<div class="chat-response">{html_with_linebreaks}</div>'
        
        return final_html
    
    def _convert_headers_to_html(self, text: str) -> str:
        """헤더를 HTML로 변환"""
        # ### Header -> <h3>Header</h3>
        text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        # ## Header -> <h2>Header</h2>
        text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        # # Header -> <h1>Header</h1>
        text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
        return text
    
    def _convert_lists_to_html(self, text: str) -> str:
        """목록을 HTML로 변환"""
        # - 또는 * 로 시작하는 줄을 <li> 태그로 변환
        lines = text.split('\n')
        result_lines = []
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            if re.match(r'^[-*]\s+(.+)', stripped):
                # 목록 항목 발견
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                # 목록 내용 추출
                list_content = re.sub(r'^[-*]\s+(.+)', r'\1', stripped)
                result_lines.append(f'<li>{list_content}</li>')
            else:
                # 목록이 아닌 줄
                if in_list:
                    result_lines.append('</ul>')
                    in_list = False
                result_lines.append(line)
        
        # 마지막에 목록이 열려있으면 닫기
        if in_list:
            result_lines.append('</ul>')
        
        return '\n'.join(result_lines)
    
    def _convert_emphasis_to_html(self, text: str) -> str:
        """강조를 HTML로 변환"""
        # **bold** -> <strong>bold</strong>
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        # *italic* -> <em>italic</em>
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        return text
    
    def _convert_linebreaks_to_html(self, text: str) -> str:
        """줄바꿈을 HTML로 변환"""
        # 연속된 줄바꿈(\n\n)을 단락으로 분리
        text = re.sub(r'\n\n+', '</p><p>', text)
        # 단일 줄바꿈(\n)을 <br>로 변환
        text = re.sub(r'\n', '<br>', text)
        
        # 전체를 <p> 태그로 감싸기 (헤더나 목록이 아닌 경우)
        if not re.match(r'^<(h[1-3]|ul)', text):
            text = f'<p>{text}</p>'
        
        # 빈 <p> 태그 제거
        text = re.sub(r'<p>\s*</p>', '', text)
        
        return text
    
    def validate_html_structure(self, html_content: str) -> Dict:
        """HTML 구조 검증"""
        validation_result = {
            "is_valid": True,
            "tag_count": 0,
            "tag_balance": True,
            "errors": []
        }
        
        try:
            # 태그 개수 계산
            tags = re.findall(r'<[^>]+>', html_content)
            validation_result["tag_count"] = len(tags)
            
            # 태그 균형 체크 (간단한 버전)
            opening_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)', html_content)
            closing_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)', html_content)
            
            # 자체 닫는 태그 제거 (br, hr 등)
            self_closing = ['br', 'hr', 'img', 'input', 'meta', 'link']
            opening_tags = [tag for tag in opening_tags if tag not in self_closing]
            
            if len(opening_tags) != len(closing_tags):
                validation_result["tag_balance"] = False
                validation_result["is_valid"] = False
                validation_result["errors"].append("태그 균형이 맞지 않습니다")
                
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"검증 중 오류: {str(e)}")
        
        return validation_result
    
    def extract_text_from_html(self, html_content: str) -> str:
        """HTML에서 텍스트만 추출"""
        # HTML 태그 제거
        text_only = re.sub(r'<[^>]+>', '', html_content)
        # HTML 엔티티 디코딩
        text_only = html.unescape(text_only)
        # 연속된 공백 및 줄바꿈 정리
        text_only = re.sub(r'\s+', ' ', text_only).strip()
        return text_only

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
    if isinstance(conversation_history, list):
        messages.extend(conversation_history)
    else:
        print(f"[ERROR] conversation_history가 리스트가 아닙니다. 타입: {type(conversation_history)}")
    
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

# ========================================================================================
# HTML 변환 API 엔드포인트 - 새로운 기능
# ========================================================================================

@router.post("/convert_html")
async def convert_text_to_html(request: Request):
    """텍스트를 HTML로 변환하는 API 엔드포인트"""
    try:
        data = await request.json()
        text_content = data.get("text", "")
        
        if not text_content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "변환할 텍스트가 비어있습니다."}
            )
        
        # HTML 변환 수행
        html_result = html_processor.convert_to_html(text_content)
        
        # HTML 구조 검증
        validation = html_processor.validate_html_structure(html_result)
        
        # 텍스트 추출 (검증용)
        extracted_text = html_processor.extract_text_from_html(html_result)
        
        # 텍스트 유사도 계산 (간단한 방식)
        original_clean = re.sub(r'\s+', ' ', text_content.strip())
        extracted_clean = re.sub(r'\s+', ' ', extracted_text.strip())
        
        similarity = 100.0
        if original_clean and extracted_clean:
            # 단순 길이 기반 유사도 (더 정확한 방법으로 대체 가능)
            min_len = min(len(original_clean), len(extracted_clean))
            max_len = max(len(original_clean), len(extracted_clean))
            similarity = (min_len / max_len) * 100 if max_len > 0 else 0
        
        # 통계 정보 생성
        stats = {
            "original_length": len(text_content),
            "html_length": len(html_result),
            "extracted_length": len(extracted_text),
            "similarity_score": round(similarity, 2),
            "tag_count": validation.get("total_tags", 0),
            "validation_errors": len(validation.get("errors", [])),
            "validation_warnings": len(validation.get("warnings", []))
        }
        
        return JSONResponse(content={
            "success": True,
            "original_text": text_content,
            "html_result": html_result,
            "extracted_text": extracted_text,
            "validation": validation,
            "stats": stats
        })
        
    except Exception as e:
        print(f"[ERROR] HTML 변환 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"HTML 변환 중 오류가 발생했습니다: {str(e)}"}
        )

@router.post("/validate_html")
async def validate_html_content(request: Request):
    """HTML 내용 검증 API 엔드포인트"""
    try:
        data = await request.json()
        html_content = data.get("html", "")
        
        if not html_content.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "검증할 HTML 내용이 비어있습니다."}
            )
        
        # HTML 구조 검증
        validation = html_processor.validate_html_structure(html_content)
        
        return JSONResponse(content={
            "success": True,
            "validation": validation
        })
        
    except Exception as e:
        print(f"[ERROR] HTML 검증 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"HTML 검증 중 오류가 발생했습니다: {str(e)}"}
        )

# ========================================================================================
# 기존 엔드포인트들
# ========================================================================================

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