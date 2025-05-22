# app/db/database.py

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# DB Engine
def get_async_engine(database_url):
    return create_async_engine(database_url)

engine = get_async_engine(DATABASE_URL)

# Session 생성기
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

from db.database import async_session, get_async_engine