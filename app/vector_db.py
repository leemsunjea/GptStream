# app/vector_db.py
import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
from db.database import async_session
from sqlalchemy import select
from db.models import documents, embeddings
from app.config import settings

load_dotenv()
openai.api_key = settings.OPENAI_API_KEY

dimension = 1536  # OpenAI 임베딩 차원
index = faiss.IndexFlatL2(dimension)
doc_store = []

async def load_faiss_and_docstore():
    embeddings_list = []
    async with async_session() as session:
        # 문서 불러오기
        docs = await session.execute(select(documents.c.content).order_by(documents.c.id))
        doc_store.extend([row[0] for row in docs.fetchall()])
        # 임베딩 불러오기
        embs = await session.execute(select(embeddings.c.embedding).order_by(embeddings.c.id))
        embeddings_list = [np.frombuffer(row[0], dtype=np.float32) for row in embs.fetchall()]
    if embeddings_list:
        index.add(np.stack(embeddings_list))
    return index, doc_store

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')