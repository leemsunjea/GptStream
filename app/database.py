from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/gptstream")

# 비동기 엔진 생성
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=True,
    future=True
)

# 비동기 세션 생성
async_session = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_async_engine():
    return async_engine

async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close() 