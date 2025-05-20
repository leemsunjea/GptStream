# app/main.py

from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import async_session
from app.routes import home, chat

app = FastAPI()

# DB 의존성
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/db-test")
async def test(db: AsyncSession = Depends(get_db)):
    result = await db.execute("SELECT 1")
    return {"db_connection": result.scalar()}

# 기존 라우터 등록
app.include_router(home.router)
app.include_router(chat.router)