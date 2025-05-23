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
    
    print(f"문서 개수: {len(doc_store)}, 임베딩 개수: {len(embeddings_list)}")
    
    if embeddings_list:
        index.add(np.stack(embeddings_list))
        print(f"FAISS 인덱스에 추가된 벡터 개수: {index.ntotal}")
    else:
        print("임베딩 데이터가 없습니다. FAISS 인덱스가 비어 있습니다.")
    return index, doc_store

async def get_embedding_async(text: str) -> Optional[np.ndarray]:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.embeddings.create(input=text, model="text-embedding-ada-002")
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        print(f"생성된 임베딩: {embedding[:5]}...")  # 일부 데이터 출력
        return embedding
    except Exception as e:
        print(f"[임베딩 오류] {e}")
        return None

def split_text_to_paragraphs(text: str) -> list:
    # 간단한 문단 분리 로직 (예시)
    return [p.strip() for p in text.split("\n\n") if p.strip()]

async def pollTaskStatus(task_id, api_base, append_system_log):
    import aiohttp
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_base}/task_status/{task_id}") as response:
                result = await response.json()
                print("Task 상태:", result)  # 디버깅 로그 추가
                if result.get("logs") and isinstance(result["logs"], list):
                    for msg in result["logs"]:
                        append_system_log(msg)
        await asyncio.sleep(5)  # 5초마다 상태 확인