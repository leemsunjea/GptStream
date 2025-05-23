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
import aiohttp

load_dotenv()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

dimension = 1536
index = faiss.IndexFlatL2(dimension)
doc_store = []

async def load_faiss_and_docstore():
    embeddings_list = []
    async with async_session() as session:
        # 문서 내용 로드
        try:
            docs = await session.execute(select(documents.c.content).order_by(documents.c.id))
            doc_store.extend([row[0] for row in docs.fetchall()])
            print(f"[DEBUG] 문서 개수: {len(doc_store)}")
            if len(doc_store) > 0:
                print(f"[DEBUG] 첫 번째 문서 내용: {doc_store[0][:100]}")  # 첫 번째 문서 일부 출력
            else:
                print("[DEBUG] 문서 데이터가 없습니다.")
        except Exception as e:
            print(f"[ERROR] 문서 로드 중 오류 발생: {e}")

        # 임베딩 데이터 로드
        try:
            embs = await session.execute(select(embeddings.c.embedding).order_by(embeddings.c.id))
            embeddings_list = [np.frombuffer(row[0], dtype=np.float32) for row in embs.fetchall()]
            print(f"[DEBUG] 임베딩 개수: {len(embeddings_list)}")
            if len(embeddings_list) > 0:
                print(f"[DEBUG] 첫 번째 임베딩 데이터: {embeddings_list[0][:5]}")  # 첫 번째 임베딩 일부 출력
            else:
                print("[DEBUG] 임베딩 데이터가 없습니다.")
        except Exception as e:
            print(f"[ERROR] 임베딩 로드 중 오류 발생: {e}")

    # FAISS 인덱스에 데이터 추가
    if embeddings_list:
        try:
            index.add(np.stack(embeddings_list))
            print(f"[DEBUG] FAISS 인덱스에 추가된 벡터 개수: {index.ntotal}")
        except Exception as e:
            print(f"[ERROR] FAISS 인덱스 추가 중 오류 발생: {e}")
    else:
        print("[DEBUG] FAISS 인덱스가 비어 있습니다. 데이터가 추가되지 않았습니다.")
    return index, doc_store

async def get_embedding_async(text: str) -> Optional[np.ndarray]:
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.embeddings.create(input=text, model="text-embedding-ada-002")
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        print(f"[DEBUG] 생성된 임베딩 데이터: {embedding[:5]}")  # 일부 데이터 출력
        return embedding
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 중 오류 발생: {e}")
        return None

def split_text_to_paragraphs(text: str) -> list:
    # 간단한 문단 분리 로직 (예시)
    return [p.strip() for p in text.split("\n\n") if p.strip()]

async def pollTaskStatus(task_id, api_base, append_system_log):

    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_base}/task_status/{task_id}") as response:
                result = await response.json()
                print("Task 상태:", result)  # 디버깅 로그 추가
                if result.get("logs") and isinstance(result["logs"], list):
                    for msg in result["logs"]:
                        append_system_log(msg)
        await asyncio.sleep(5)  # 5초마다 상태 확인

# The following block should be inside a function, not at the module level.
# Example: wrap it in an async function for later use.

async def search_similar_documents(message):
    if index.ntotal > 0:
        query_embedding = await get_embedding_async(message)
        if query_embedding is None:
            print("[ERROR] 쿼리 임베딩 생성 실패")
            return None
        
        print(f"[DEBUG] FAISS 인덱스 벡터 개수: {index.ntotal}")
        try:
            D, I = index.search(np.array([query_embedding]), k=7)  # 상위 7개 문서 검색
            print(f"[DEBUG] 검색 결과 인덱스: {I}, 거리: {D}")
        except Exception as e:
            print(f"[ERROR] FAISS 검색 중 오류 발생: {e}")
            return None
        
        if I is not None and len(I[0]) > 0:
            referenced_docs = [doc_store[i] for i in I[0] if i >= 0 and i < len(doc_store)]
            print(f"[DEBUG] 검색된 문서 개수: {len(referenced_docs)}")
            if len(referenced_docs) > 0:
                print(f"[DEBUG] 첫 번째 검색된 문서 내용: {referenced_docs[0][:100]}")  # 첫 번째 문서 일부 출력
            else:
                print("[DEBUG] 검색된 문서가 없습니다.")
            return referenced_docs
        else:
            print("[DEBUG] 검색 결과가 없습니다.")
            return []
    else:
        print("[DEBUG] 인덱스에 데이터가 없습니다.")
        return []