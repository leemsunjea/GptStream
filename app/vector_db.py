# app/vector_db.py
import os
import faiss
import numpy as np
import aiohttp # aiohttp 임포트 추가
from dotenv import load_dotenv
from db.database import async_session
from sqlalchemy import select, text # sqlalchemy 임포트 확인
from db.models import documents, embeddings
from app.config import settings
from openai import OpenAI
from typing import Optional
import asyncio

load_dotenv()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

dimension = 1536
index = faiss.IndexFlatL2(dimension)
doc_store = []  # 전역 변수로 선언된 문서 저장소

async def load_faiss_and_docstore():
    global doc_store, index
    print("[INFO] load_faiss_and_docstore: FAISS 인덱스 및 문서 저장소 로드 시작...")
    new_doc_store = []
    embedding_vectors = []

    async with async_session() as session:
        # 모든 문서 및 해당 user_id, title, summary, response_style, created_at, pdf_name 가져오기
        stmt_docs = select(
            documents.c.id, 
            documents.c.user_id, 
            documents.c.content,
            documents.c.title,
            documents.c.summary,
            documents.c.response_style,
            documents.c.created_at,
            documents.c.pdf_name
        ).order_by(documents.c.id)
        result_docs = await session.execute(stmt_docs)
        all_db_documents = result_docs.fetchall()

        print(f"[INFO] load_faiss_and_docstore: DB에서 {len(all_db_documents)}개의 문서 로드 완료.")
        if not all_db_documents:
            print("[INFO] load_faiss_and_docstore: 데이터베이스에 문서가 없습니다. doc_store와 index를 초기화합니다.")
            doc_store = []
            if index.ntotal > 0: # 문서가 없고 인덱스에 데이터가 남아있다면 인덱스 초기화
                index.reset()
                print("[INFO] load_faiss_and_docstore: 기존 FAISS 인덱스 초기화 완료.")
            return index, doc_store

        # 각 문서에 대해 문단으로 분할하고 해당 임베딩 가져오기
        for doc_id, user_id, page_content, title, summary, response_style, created_at, pdf_name in all_db_documents:
            print(f"[DEBUG] load_faiss_and_docstore: 문서 처리 중 - doc_id: {doc_id}, user_id: {user_id}, title: {title}")
            if not page_content: # 페이지 내용이 비어있다면 건너뜀
                print(f"[DEBUG] load_faiss_and_docstore: doc_id {doc_id} (user_id: {user_id})은(는) 페이지 내용이 비어있어 건너뜁니다.")
                continue
            paragraphs = split_text_to_paragraphs(page_content)

            # 문단에 대한 임베딩 가져오기
            stmt_embs = select(embeddings.c.embedding).where(embeddings.c.document_id == doc_id).order_by(embeddings.c.id)
            result_embs = await session.execute(stmt_embs)
            paragraph_embeddings_bytes = [row[0] for row in result_embs.fetchall()]

            print(f"[DEBUG] load_faiss_and_docstore: doc_id {doc_id} (user_id: {user_id}) - 문단 수: {len(paragraphs)}, DB 임베딩 수: {len(paragraph_embeddings_bytes)}")

            if len(paragraphs) != len(paragraph_embeddings_bytes):
                print(f"[WARNING] load_faiss_and_docstore: doc_id {doc_id} (user_id: {user_id})에 대해 문단 수({len(paragraphs)})와 임베딩 수({len(paragraph_embeddings_bytes)}) 불일치. 해당 문서는 건너뜁니다.")
                continue

            for i, para_text in enumerate(paragraphs):
                new_doc_store.append({
                    'text': para_text,
                    'user_id': user_id,
                    'doc_id': doc_id,
                    'title': title,
                    'summary': summary,
                    'response_style': response_style,
                    'created_at': created_at.isoformat() if created_at else None, # ISO 형식으로 변환
                    'pdf_name': pdf_name
                })
                embedding_bytes = paragraph_embeddings_bytes[i]
                embedding_vectors.append(np.frombuffer(embedding_bytes, dtype=np.float32))

    if new_doc_store and embedding_vectors:
        doc_store = new_doc_store # 전역 doc_store에 할당
        print(f"[INFO] load_faiss_and_docstore: 전역 doc_store 업데이트 완료. 총 {len(doc_store)}개 항목.")
        if doc_store:
            print(f"[DEBUG] load_faiss_and_docstore: 업데이트된 doc_store의 첫 번째 항목: user_id={doc_store[0].get('user_id')}, doc_id={doc_store[0].get('doc_id')}, title='{doc_store[0].get('title')}', text='{doc_store[0].get('text', '')[:30]}...'")

        index.reset() # 기존 인덱스 초기화
        print(f"[INFO] load_faiss_and_docstore: FAISS 인덱스 초기화 완료. 추가할 벡터 수: {len(embedding_vectors)}")
        index.add(np.stack(embedding_vectors))
        print(f"[INFO] load_faiss_and_docstore: FAISS 인덱스에 {index.ntotal} 벡터 로드 완료. 문서 저장소 크기: {len(doc_store)}")
    else:
        doc_store = []
        if index.ntotal > 0:
            index.reset()
        print("[INFO] load_faiss_and_docstore: FAISS 인덱스에 임베딩이 로드되지 않았습니다. new_doc_store 또는 embedding_vectors가 비어있습니다. doc_store와 index를 초기화합니다.")

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

async def search_similar_documents(message: str, user_id: str): # user_id 매개변수 추가
    print(f"[INFO] search_similar_documents: 검색 시작 - 사용자: {user_id}, 메시지: '{message[:50]}...'")
    if index.ntotal == 0:
        print(f"[DEBUG] search_similar_documents: 인덱스가 비어 있습니다 (index.ntotal: {index.ntotal}). 사용자 {user_id}에 대한 검색 문서가 없습니다.")
        return []

    query_embedding = await get_embedding_async(message)
    if query_embedding is None:
        print(f"[ERROR] search_similar_documents: 쿼리 임베딩 생성 실패: 사용자 {user_id}.")
        return []

    print(f"[DEBUG] search_similar_documents: FAISS 인덱스 검색 (총 크기: {index.ntotal}) - 사용자: {user_id}")
    try:
        k_search = min(index.ntotal, 7) 
        print(f"[DEBUG] search_similar_documents: k_search 값: {k_search}")
        if k_search == 0 : 
             print(f"[DEBUG] search_similar_documents: 인덱스가 사실상 비어있음 (k_search=0). 사용자 {user_id}.")
             return []

        D, I = index.search(np.array([query_embedding]), k=k_search)
        print(f"[DEBUG] search_similar_documents: FAISS 검색 원본 결과 - 사용자 {user_id} - 인덱스: {I}, 거리: {D}")
    except Exception as e:
        print(f"[ERROR] search_similar_documents: FAISS 검색 오류 - 사용자 {user_id}: {e}")
        return []

    # referenced_docs_content 대신 referenced_docs_details 로 변경하여 메타데이터 포함
    referenced_docs_details = [] 
    if I is not None and len(I[0]) > 0:
        print(f"[DEBUG] search_similar_documents: FAISS 결과 {len(I[0])}개 항목 필터링 시작 - 사용자: {user_id}")
        unique_doc_ids = set() # 중복된 doc_id의 전체 메타데이터 반환 방지

        for rank, doc_index in enumerate(I[0]):
            print(f"[DEBUG] search_similar_documents: 필터링 중 - rank: {rank}, doc_index: {doc_index}")
            if 0 <= doc_index < len(doc_store): 
                item = doc_store[doc_index]
                # user_id로 필터링하고, 아직 추가되지 않은 doc_id인 경우에만 추가
                if isinstance(item, dict) and item.get('user_id') == user_id:
                    # 검색 결과에는 문단 텍스트와 함께 전체 문서의 메타데이터를 포함시킬 수 있음
                    # 여기서는 검색된 문단(item) 자체를 반환 (이미 메타데이터 포함)
                    referenced_docs_details.append(item) 
                    print(f"[DEBUG] search_similar_documents: 사용자 {user_id} 문서 일치! referenced_docs_details에 추가됨: {item.get('doc_id')}, {item.get('title')}")
                else:
                    print(f"[DEBUG] search_similar_documents: 사용자 {user_id} 문서 불일치 (doc_store user_id: {item.get('user_id')}).")
            else:
                print(f"[WARNING] search_similar_documents: FAISS 검색 결과의 잘못된 인덱스 {doc_index} (doc_store 크기: {len(doc_store)}) - 사용자 {user_id}.")
        
        print(f"[DEBUG] search_similar_documents: 사용자 {user_id}에 대해 {len(I[0])}개의 원본 결과에서 {len(referenced_docs_details)}개의 상세 정보가 최종 필터링됨.")
    else:
        print(f"[DEBUG] search_similar_documents: 사용자 {user_id}에 대한 FAISS 검색 결과가 없습니다 (I is None or 비어있음).")
    
    return referenced_docs_details # 상세 정보 반환