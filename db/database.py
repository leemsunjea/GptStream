# app/db/database.py
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Optional
from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# 로깅 설정
logger = logging.getLogger(__name__)

# 기본 데이터 디렉토리 설정 (Docker 환경 고려)
DEFAULT_DATA_DIR = "/tmp/gptstream"
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

# 세션 ID를 위한 쿠키 이름
SESSION_COOKIE_NAME = "gptstream_session"

def get_user_data_dir(session_id: str) -> str:
    """세션 ID 기반 사용자 데이터 디렉토리 경로 반환"""
    user_dir = os.path.join(DATA_DIR, "users", session_id)
    try:
        Path(user_dir).mkdir(parents=True, exist_ok=True)
        return user_dir
    except Exception as e:
        logger.error(f"사용자 디렉토리 생성 실패: {e}")
        raise HTTPException(status_code=500, detail="사용자 저장소를 초기화할 수 없습니다.")

def get_db_path(session_id: str) -> str:
    """세션 ID 기반 데이터베이스 파일 경로 반환"""
    user_dir = get_user_data_dir(session_id)
    return os.path.join(user_dir, "app.db")

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
async def get_async_session(session_id: str):
    """세션 ID에 해당하는 비동기 세션 반환"""
    engine = get_db_engine(session_id)
    return sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False
    )

# 이전 버전 호환성을 위한 기본 세션 (사용자 지정이 필요 없는 경우 사용)
# 주의: 이 엔진은 실제로는 사용되지 않으며, get_async_session()을 통해 생성된 세션을 사용해야 함
_engine = get_db_engine("default")
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

# 의존성 주입을 위한 세션 제너레이터
async def get_db():
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

from db.database import async_session, get_async_engine