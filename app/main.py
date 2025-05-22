# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session
from app.routes import home, chat
from app.routes import vector  # 추가
from app.vector_db import load_faiss_and_docstore
from fastapi import FastAPI, HTTPException
from app.routes.vector import router
from db.models import documents, embeddings
import asyncio

app = FastAPI()

app.include_router(router)

# 서버 실행 시 초기화
@app.on_event("startup")
async def on_startup():
    async with async_session() as session:
        async with session.begin():
            await session.execute(documents.delete())
            await session.execute(embeddings.delete())
    print("✅ DB 초기화 완료")

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