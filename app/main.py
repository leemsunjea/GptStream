# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session
from app.routes import home, chat
from app.routes import vector  # 추가
from app.vector_db import load_faiss_and_docstore

app = FastAPI()

# DB 의존성
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/db-test")
async def test(db: AsyncSession = Depends(get_db)):
    result = await db.execute("SELECT 1")
    return {"db_connection": result.scalar()}

@app.on_event("startup")
async def startup_event():
    global index, doc_store
    index, doc_store = await load_faiss_and_docstore()

# 기존 라우터 등록
app.include_router(home.router)
app.include_router(chat.router)
app.include_router(vector.router)  # 추가