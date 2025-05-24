# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session, engine, Base  # engine, Base 추가
from app.routes import home, chat
from app.routes import vector  # 추가
from app.vector_db import load_faiss_and_docstore
from fastapi import FastAPI, HTTPException
from app.routes.vector import router
from db.models import documents, embeddings, UserPreference  # UserPreference 모델 임포트
from sqlalchemy import select, func  # select, func 임포트
from fastapi.responses import JSONResponse
import asyncio

app = FastAPI()

app.include_router(router)

# 서버 실행 시 초기화
@app.on_event("startup")
async def on_startup():
    async with async_session() as session:
        async with session.begin():
            await session.execute(embeddings.delete())            
            await session.execute(documents.delete())
    await load_faiss_and_docstore()
    print("✅ DB 초기화 완료")
    print("✅ FAISS 인덱스 및 문서 저장소 초기화 완료")

# 기본 시스템 프롬프트 목록
DEFAULT_SYSTEM_PROMPTS = [
    "You are a friendly and helpful AI assistant.",
    "You are a professional AI assistant that provides concise and accurate information.",
    "You are a creative AI assistant that can help with brainstorming and writing."
]

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # 개발 중 테이블 재생성이 필요할 때 주석 해제
        await conn.run_sync(Base.metadata.create_all)
        print("데이터베이스 테이블 생성 완료")

    # 기본 시스템 프롬프트 추가
    async with async_session() as session:
        for prompt_text in DEFAULT_SYSTEM_PROMPTS:
            # 이미 해당 프롬프트가 있는지 확인 (user_id='default_system' 및 내용으로)
            existing_prompt_result = await session.execute(
                select(UserPreference).where(
                    UserPreference.user_id == "default_system",
                    UserPreference.system_prompt == prompt_text
                )
            )
            existing_prompt = existing_prompt_result.scalars().first()

            if not existing_prompt:
                default_prompt = UserPreference(
                    user_id="default_system", 
                    system_prompt=prompt_text
                )
                session.add(default_prompt)
                print(f"기본 시스템 프롬프트 추가: '{prompt_text}'")
        await session.commit()

# DB 의존성
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/db-test")
async def test(db: AsyncSession = Depends(get_db)):
    result = await db.execute("SELECT 1")
    return {"db_connection": result.scalar()}

@app.exception_handler(504)
async def gateway_timeout_handler(request, exc):
  return JSONResponse(
    status_code=504,
    content={"success": False, "logs": ["서버 타임아웃 발생"], "detail": "Gateway Timeout"}
  )

# 기존 라우터 등록
app.include_router(home.router)
app.include_router(chat.router)
app.include_router(vector.router)  # 추가