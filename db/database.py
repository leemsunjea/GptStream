# app/db/database.py
import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# 기본 데이터 디렉토리 설정 (Docker 환경 고려)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))

# 데이터베이스 파일 경로 설정
DB_PATH = os.path.join(DATA_DIR, "app.db")

# 디렉토리 생성
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# SQLite URL (aiosqlite 사용)
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

# 비동기 엔진 설정
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # 개발 시 SQL 로그 출력
    poolclass=NullPool,  # SQLite는 connection pooling이 필요 없음
    connect_args={
        "check_same_thread": False,  # SQLite 비동기 작업을 위한 설정
        "timeout": 30,  # 타임아웃 설정 (초)
        "isolation_level": "AUTOCOMMIT"  # 자동 커밋 설정
    }
)

# 비동기 세션 팩토리
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 베이스 모델
Base = declarative_base()

# 데이터베이스 초기화 함수 (앱 시작 시 호출)
async def init_db():
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