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

    # 기본 시스템 프롬프트 추가 (단일 항목만 보장)
    async with async_session() as session:
        # 'default_system' ID로 이미 프롬프트가 있는지 확인
        stmt = select(UserPreference).where(UserPreference.user_id == "default_system")
        result = await session.execute(stmt)
        existing_default_user_pref = result.scalars().first()

        if not existing_default_user_pref:
            if DEFAULT_SYSTEM_PROMPTS: # 목록에 프롬프트가 있는지 확인
                # 첫 번째 프롬프트를 기본값으로 추가
                default_prompt_entry = UserPreference(
                    user_id="default_system",
                    system_prompt=DEFAULT_SYSTEM_PROMPTS[0] # 첫 번째 프롬프트를 사용
                )
                session.add(default_prompt_entry)
                await session.commit()
                print(f"기본 시스템 프롬프트 ('{DEFAULT_SYSTEM_PROMPTS[0]}') 추가 완료.")
            else:
                print("DEFAULT_SYSTEM_PROMPTS 목록이 비어 있어, 기본 시스템 프롬프트를 추가할 수 없습니다.")
        else:
            print(f"기존 기본 시스템 프롬프트 ('{existing_default_user_pref.system_prompt}')가 이미 존재합니다. 추가 작업을 건너뜁니다.")

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