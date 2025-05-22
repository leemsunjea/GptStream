import os
import faiss
import numpy as np
from dotenv import load_dotenv
from db.database import async_session
from sqlalchemy import select
from db.models import documents, embeddings
from app.config import settings
from openai import OpenAI
from typing import Optional
import asyncio

load_dotenv()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

dimension = 1536
index = faiss.IndexFlatL2(dimension)
doc_store = []

async def load_faiss_and_docstore():
    embeddings_list = []
    async with async_session() as session:
        docs = await session.execute(select(documents.c.content).order_by(documents.c.id))
        doc_store.extend([row[0] for row in docs.fetchall()])
        embs = await session.execute(select(embeddings.c.embedding).order_by(embeddings.c.id))
        embeddings_list = [np.frombuffer(row[0], dtype=np.float32) for row in embs.fetchall()]
    if embeddings_list:
        index.add(np.stack(embeddings_list))
    return index, doc_store

async def get_embedding_async(text: str) -> Optional[np.ndarray]:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.embeddings.create(input=text, model="text-embedding-ada-002")
        )
        return np.array(response.data[0].embedding, dtype='float32')
    except Exception as e:
        print(f"[임베딩 오류] {e}")
        return None

def split_text_to_paragraphs(text: str) -> list:
    # 간단한 문단 분리 로직 (예시)
    return [p.strip() for p in text.split("\n\n") if p.strip()]