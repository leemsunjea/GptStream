# app/db/database.py
import os
import sys
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 데이터 디렉토리 설정 (Docker 환경 고려)
# 컨테이너 내부에서 /app/data를 사용 (Docker 볼륨으로 마운트 권장)
DEFAULT_DATA_DIR = "/app/data"
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

# 전역 변수로 세션별 DB 엔진 캐시
_session_engines: Dict[str, Any] = {}
_session_locks: Dict[str, Any] = {}

# 필요한 디렉토리 생성
try:
    os.makedirs(DATA_DIR, exist_ok=True, mode=0o777)  # 모든 사용자에게 쓰기 권한 부여
except Exception as e:
    logger.error(f"기본 데이터 디렉토리 생성 실패: {e}")
    # 기본 디렉토리 생성 실패 시 임시 디렉토리 사용
    DATA_DIR = os.path.join(tempfile.gettempdir(), "gptstream")
    os.makedirs(DATA_DIR, exist_ok=True, mode=0o777)

# 세션 ID를 위한 쿠키 이름
SESSION_COOKIE_NAME = "gptstream_session"

def ensure_directory(path: str) -> None:
    """디렉토리가 존재하는지 확인하고 없으면 생성"""
    try:
        os.makedirs(path, exist_ok=True, mode=0o777)
        os.chmod(path, 0o777)
    except Exception as e:
        logger.error(f"디렉토리 생성 실패 {path}: {e}")
        raise

def get_user_data_dir(session_id: str) -> str:
    """세션 ID 기반 사용자 데이터 디렉토리 경로 반환"""
    if not session_id:
        raise ValueError("session_id는 필수입니다.")
    
    # 기본 사용자 디렉토리 경로
    user_dir = os.path.join(DATA_DIR, "users", session_id)
    
    try:
        # 메인 사용자 디렉토리 생성
        ensure_directory(user_dir)
        
        # 필요한 하위 디렉토리들 생성
        for subdir in ['uploads', 'vector_index', 'db']:
            subdir_path = os.path.join(user_dir, subdir)
            ensure_directory(subdir_path)
            
        return user_dir
        
    except Exception as e:
        logger.error(f"사용자 디렉토리 생성 실패: {e}")
        # 실패 시 임시 디렉토리 사용
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), "gptstream", "users", session_id)
            ensure_directory(temp_dir)
            
            # 임시 디렉토리 내에 하위 디렉토리들 생성
            for subdir in ['uploads', 'vector_index', 'db']:
                ensure_directory(os.path.join(temp_dir, subdir))
                
            logger.warning(f"임시 디렉토리 사용: {temp_dir}")
            return temp_dir
            
        except Exception as temp_e:
            logger.critical(f"임시 디렉토리 생성에도 실패: {temp_e}")
            # 최종적으로 메모리 기반 임시 디렉토리 사용
            temp_dir = os.path.join("/dev/shm", "gptstream", "users", session_id)
            ensure_directory(temp_dir)
            return temp_dir

def get_db_path(session_id: str) -> str:
    """세션 ID 기반 데이터베이스 파일 경로 반환"""
    user_dir = get_user_data_dir(session_id)
    return os.path.join(user_dir, "db", f"{session_id}.db")

def get_or_create_session_id(request: Request) -> str:
    """요청에서 세션 ID를 가져오거나 새로 생성"""
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = str(uuid.uuid4())
        request.cookies[SESSION_COOKIE_NAME] = session_id
    return session_id

def get_db_engine(session_id: str):
    """세션 ID에 해당하는 데이터베이스 엔진 반환"""
    db_path = get_db_path(session_id)
    db_url = f"sqlite+aiosqlite:///{db_path}"
    
    return create_async_engine(
        db_url,
        echo=True,
        poolclass=NullPool,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
            "isolation_level": "AUTOCOMMIT"
        }
    )

# 비동기 세션 팩토리 생성 함수
async def get_async_session(session_id: str = "default") -> AsyncSession:
    """세션 ID에 해당하는 비동기 세션 반환"""
    # 세션 ID가 없으면 기본값 사용
    if not session_id or session_id == "default":
        session_id = str(uuid.uuid4())
        
    # 세션별로 고유한 DB 엔진 생성
    if session_id not in _session_engines:
        _session_engines[session_id] = get_db_engine(session_id)
        _session_locks[session_id] = asyncio.Lock()
    
    # 세션 팩토리 생성
    async_session_factory = sessionmaker(
        _session_engines[session_id], 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    return async_session_factory()

# 이전 버전 호환성을 위한 기본 세션 (사용자 지정이 필요 없는 경우 사용)
# 주의: 이 엔진은 실제로는 사용되지 않으며, get_async_session()을 통해 생성된 세션을 사용해야 함
_engine = get_db_engine("__system__")
async_session = sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 베이스 모델
Base = declarative_base()

# 데이터베이스 초기화 함수 (앱 시작 시 호출)
async def init_db(session_id: str = "default"):
    """지정된 세션 ID의 데이터베이스 초기화"""
    engine = get_db_engine(session_id)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# 의존성 주입을 위한 DB 세션 생성기
async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """의존성 주입을 위한 DB 세션 생성기"""
    session_id = get_or_create_session_id(request)
    
    # 세션별 락 획득 (동시성 제어)
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    
    async with _session_locks[session_id]:
        async with get_async_session(session_id) as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database error: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"데이터베이스 오류가 발생했습니다: {str(e)}"
                )
            finally:
                await session.close()
from db.database import async_session, get_async_engine