# app/routes/chat.py

import os
import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
import openai
from openai import OpenAI, AsyncOpenAI
from app.config import settings
from db.models import ChatHistory
from db.database import get_async_session, get_or_create_session_id, get_db, get_user_data_dir
from fastapi import HTTPException, status

# 로깅 설정
logger = logging.getLogger(__name__)

# OpenAI 클라이언트 초기화
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# 채팅 모델 설정
CHAT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# 세션별 채팅 기록을 저장할 기본 디렉토리
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")

# 디렉토리 생성
try:
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True, mode=0o777)
except Exception as e:
    logger.error(f"채팅 기록 디렉토리 생성 실패: {e}")
    # 실패 시 임시 디렉토리 사용
    CHAT_HISTORY_DIR = os.path.join(tempfile.gettempdir(), "gptstream", "chat_history")
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True, mode=0o777)

def get_chat_history_path(session_id: str) -> str:
    """세션 ID에 해당하는 채팅 기록 파일 경로를 반환합니다."""
    try:
        user_dir = get_user_data_dir(session_id)
        chat_dir = os.path.join(user_dir, "chat_history")
        os.makedirs(chat_dir, exist_ok=True, mode=0o777)
        return os.path.join(chat_dir, "chat.json")
    except Exception as e:
        logger.error(f"채팅 기록 경로 생성 실패: {e}")
        # 실패 시 임시 디렉토리 사용
        temp_dir = os.path.join(tempfile.gettempdir(), "gptstream", "chat_history", session_id)
        os.makedirs(temp_dir, exist_ok=True, mode=0o777)
        return os.path.join(temp_dir, "chat.json")

def get_or_create_session_store(session_id: str) -> dict:
    """세션 ID에 해당하는 저장소를 가져오거나 생성합니다."""
    session_store = {
        "messages": [],
        "documents": [],
        "index": None,
        "doc_store": {},
        "created_at": time.time(),
        "last_accessed": time.time()
    }
    
    # 채팅 기록 파일 경로
    history_file = get_chat_history_path(session_id)
    
    try:
        # 기존 채팅 기록이 있으면 로드
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                # 필요한 필드만 업데이트 (보안상 안전하게)
                for key in ["messages", "documents", "doc_store"]:
                    if key in saved_data:
                        session_store[key] = saved_data[key]
                
                # 메타데이터 유지
                if "created_at" in saved_data:
                    session_store["created_at"] = saved_data["created_at"]
    
    except Exception as e:
        logging.error(f"채팅 기록 로드 실패: {e}")
        # 오류 발생 시 새 세션 생성
    
    return session_store

async def load_chat_history(session_id: str, max_messages: int = 20) -> List[Dict[str, str]]:
    """세션 ID에 해당하는 채팅 기록을 로드합니다.
    
    Args:
        session_id: 사용자 세션 ID
        max_messages: 최대 로드할 메시지 수 (가장 최근 메시지 기준)
    """
    chat_history_path = get_chat_history_path(session_id)
    if os.path.exists(chat_history_path):
        try:
            with open(chat_history_path, 'r', encoding='utf-8') as f:
                messages = json.load(f)
                # 최신 메시지 순으로 정렬 후 최대 개수만큼 반환
                messages.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
                return messages[:max_messages]
        except Exception as e:
            logger.error(f"채팅 기록 로드 실패: {e}")
            # 손상된 파일 백업
            try:
                backup_path = f"{chat_history_path}.bak.{int(time.time())}"
                shutil.copy2(chat_history_path, backup_path)
                logger.info(f"손상된 채팅 기록 백업: {backup_path}")
            except Exception as backup_error:
                logger.error(f"채팅 기록 백업 실패: {backup_error}")
    return []

async def save_chat_history(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """채팅 기록을 파일에 저장합니다.
    
    Args:
        session_id: 사용자 세션 ID
        messages: 저장할 메시지 목록
    """
    if not messages:
        return
        
    chat_history_path = get_chat_history_path(session_id)
    temp_path = f"{chat_history_path}.tmp"
    
    try:
        # 기존 파일이 있으면 백업
        if os.path.exists(chat_history_path):
            backup_path = f"{chat_history_path}.bak"
            shutil.copy2(chat_history_path, backup_path)
        
        # 임시 파일에 저장
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2, ensure_ascii=False)
        
        # 임시 파일을 원본으로 이동 (원자적 연산)
        shutil.move(temp_path, chat_history_path)
        
        # 파일 권한 설정
        os.chmod(chat_history_path, 0o666)
        
    except Exception as e:
        logger.error(f"채팅 기록 저장 실패: {e}")
        # 임시 파일이 남아있으면 삭제
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=settings.OPENAI_API_KEY)

async def get_embedding_async(text: str) -> Optional[List[float]]:
    """텍스트를 임베딩으로 변환"""
    try:
        response = await asyncio.to_thread(
            openai.Embedding.create,
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 실패: {e}")
        return None

@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    db = Depends(get_db)
):
    """채팅 메시지를 처리하고 스트리밍 응답을 반환합니다."""
    try:
        # 채팅 기록 로드 (최근 10개 메시지만 로드)
        chat_history = await load_chat_history(session_id, max_messages=10)
        
        # 시스템 프롬프트 설정 (없는 경우에만 추가)
        if not any(msg.get("role") == "system" for msg in chat_history):
            system_prompt = {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer concisely and helpfully.",
                "timestamp": int(time.time())
            }
            chat_history.insert(0, system_prompt)
        
        # 사용자 메시지 추가
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": int(time.time())
        }
        chat_history.append(user_message)
        
        # 채팅 기록 저장 (비동기로 실행)
        asyncio.create_task(save_chat_history(session_id, chat_history))
        
        # DB에 채팅 기록 저장 (비동기로 실행)
        try:
            chat_record = ChatHistory(
                session_id=session_id,
                role="user",
                content=message,
                metadata=json.dumps({"source": "web"})
            )
            db.add(chat_record)
            db.commit()
        except Exception as db_error:
            logger.error(f"DB 저장 실패: {db_error}")
            db.rollback()
        
        # OpenAI API 호출
        response = await openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": msg["role"], "content": msg["content"]} 
                for msg in chat_history
            ],
            stream=True,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        # 스트리밍 응답 생성
        async def generate():
            assistant_message = []
            
            # 응답 스트리밍
            async for chunk in response:
                if not chunk.choices:
                    continue
                    
                content = chunk.choices[0].delta.content or ""
                if content:
                    assistant_message.append(content)
                    # SSE 형식으로 스트리밍
                    yield f"data: {json.dumps({'message': content})}\n\n"
            
            # 전체 응답을 채팅 기록에 추가
            full_response = ''.join(assistant_message)
            chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": int(time.time())
            })
            
            # 채팅 기록 저장 (비동기로 실행)
            asyncio.create_task(save_chat_history(session_id, chat_history))
            
            # DB에 어시스턴트 응답 저장 (비동기로 실행)
            try:
                assistant_record = ChatHistory(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata=json.dumps({"source": "openai"})
                )
                db.add(assistant_record)
                db.commit()
            except Exception as db_error:
                logger.error(f"어시스턴트 응답 DB 저장 실패: {db_error}")
                db.rollback()
            
            # 종료 이벤트 전송
            yield "event: end\ndata: {}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'  # Nginx 버퍼링 비활성화
            }
        )
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}", exc_info=True)
        error_detail = {
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"채팅 처리 중 오류가 발생했습니다: {str(e)}",
            headers={"X-Error-Detail": json.dumps(error_detail, ensure_ascii=False)}
        )

async def save_chat_history(session_id: str, messages: List[Dict[str, Any]]) -> None:
    """채팅 기록을 파일에 저장합니다.
    
    Args:
        session_id: 사용자 세션 ID
        messages: 저장할 메시지 목록 (role, content, timestamp 포함)
    """
    if not messages:
        return
        
    chat_history_path = get_chat_history_path(session_id)
    temp_path = f"{chat_history_path}.tmp"
    
    try:
        # 기존 파일이 있으면 백업
        if os.path.exists(chat_history_path):
            backup_path = f"{chat_history_path}.bak"
            shutil.copy2(chat_history_path, backup_path)
        
        # 임시 파일에 저장
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        
        # 임시 파일을 원본으로 이동 (원자적 연산)
        shutil.move(temp_path, chat_history_path)
        
        # 파일 권한 설정
        os.chmod(chat_history_path, 0o666)
        
    except Exception as e:
        logger.error(f"채팅 기록 저장 실패: {e}")
        # 임시 파일이 남아있으면 삭제
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise

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