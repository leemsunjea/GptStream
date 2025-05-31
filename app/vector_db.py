# app/vector_db.py

import os
import faiss
import numpy as np
import asyncio
import fitz
from openai import OpenAI
from sqlalchemy import select
from db.database import async_session
from db.models import documents, embeddings
from app.config import settings
from typing import Optional, List, Dict, Any

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# 전역 변수
dimension = 1536
index = faiss.IndexFlatL2(dimension)
doc_store = []
task_statuses = {}

async def load_faiss_and_docstore():
    """FAISS 인덱스와 문서 저장소를 로드"""
    global doc_store, index
    print("[INFO] FAISS 인덱스 및 문서 저장소 로드 시작...")
    
    new_doc_store = []
    embedding_vectors = []

    async with async_session() as session:
        # 모든 문서 조회
        stmt = select(
            documents.c.id, documents.c.user_id, documents.c.content,
            documents.c.title, documents.c.summary, documents.c.response_style,
            documents.c.created_at, documents.c.pdf_name
        ).order_by(documents.c.id)
        
        result = await session.execute(stmt)
        all_documents = result.fetchall()

        if not all_documents:
            print("[INFO] 데이터베이스에 문서가 없습니다.")
            doc_store = []
            index.reset()
            return

        # 각 문서 처리
        for doc_id, user_id, content, title, summary, response_style, created_at, pdf_name in all_documents:
            if not content:
                continue
                
            paragraphs = split_text_to_paragraphs(content)
            
            # 임베딩 조회
            emb_stmt = select(embeddings.c.embedding).where(embeddings.c.document_id == doc_id)
            emb_result = await session.execute(emb_stmt)
            emb_data = [row[0] for row in emb_result.fetchall()]

            if len(paragraphs) != len(emb_data):
                continue

            # 문서 저장소에 추가
            for i, para_text in enumerate(paragraphs):
                new_doc_store.append({
                    'text': para_text,
                    'user_id': user_id,
                    'doc_id': doc_id,
                    'title': title,
                    'summary': summary,
                    'response_style': response_style,
                    'created_at': created_at.isoformat() if created_at else None,
                    'pdf_name': pdf_name
                })
                embedding_vectors.append(np.frombuffer(emb_data[i], dtype=np.float32))

    # 인덱스 업데이트
    if new_doc_store and embedding_vectors:
        doc_store = new_doc_store
        index.reset()
        index.add(np.stack(embedding_vectors))
        print(f"[INFO] FAISS 인덱스 로드 완료: {index.ntotal}개 벡터")
    else:
        doc_store = []
        index.reset()
        print("[INFO] 인덱스 초기화 완료")

async def get_embedding_async(text: str) -> Optional[np.ndarray]:
    """텍스트의 임베딩 생성"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.embeddings.create(input=text, model="text-embedding-ada-002")
        )
        return np.array(response.data[0].embedding, dtype='float32')
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 실패: {e}")
        return None

def split_text_to_paragraphs(text: str) -> List[str]:
    """텍스트를 문단으로 분할"""
    paragraphs = []
    basic_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    for paragraph in basic_paragraphs:
        # 목차 관련 내용은 보존
        is_toc = any(keyword in paragraph.lower() for keyword in ["목차", "차례", "contents"])
        
        if len(paragraph) > 1000 and not is_toc:
            # 긴 문단 분할
            sentences = paragraph.split('. ')
            chunk = ""
            for sentence in sentences:
                if len(chunk + sentence) < 800:
                    chunk += sentence + ". "
                else:
                    if chunk.strip():
                        paragraphs.append(chunk.strip())
                    chunk = sentence + ". "
            if chunk.strip():
                paragraphs.append(chunk.strip())
        else:
            paragraphs.append(paragraph)
    
    return paragraphs

async def search_similar_documents(message: str, user_id: str) -> List[Dict[str, Any]]:
    """유사 문서 검색"""
    if index.ntotal == 0:
        return []

    # 목차 관련 검색 확장
    is_toc_query = any(keyword in message for keyword in ["목차", "차례", "내용"])
    search_query = f"{message} 목차 차례 구성" if is_toc_query else message

    query_embedding = await get_embedding_async(search_query)
    if query_embedding is None:
        return []

    try:
        k = min(index.ntotal, 15 if is_toc_query else 7)
        D, I = index.search(np.array([query_embedding]), k=k)
    except Exception as e:
        print(f"[ERROR] 검색 실패: {e}")
        return []

    results = []
    for doc_index in I[0]:
        if 0 <= doc_index < len(doc_store):
            item = doc_store[doc_index]
            if item.get('user_id') == user_id:
                results.append(item)

    return results

async def search_recent_documents_first(query: str, user_id: str) -> List[Dict[str, Any]]:
    """최근 문서 우선 검색"""
    all_results = await search_similar_documents(query, user_id)
    if not all_results:
        return []

    try:
        # 최신순 정렬
        sorted_results = sorted(
            all_results, 
            key=lambda x: x.get('created_at', x.get('doc_id', 0)), 
            reverse=True
        )
        return sorted_results[:5]
    except:
        return all_results[:5]

async def process_pdf(task_id: str, file_path: str, filename: str, session_factory, logs: List[str], user_id: str):
    """PDF 처리 함수"""
    print(f"[INFO] PDF 처리 시작: {filename}")
    
    # 상태 초기화
    task_statuses[task_id] = {
        "status": "processing", 
        "logs": logs, 
        "page_count": 0, 
        "filename": filename
    }
    
    try:
        async with session_factory() as session:
            async with session.begin():
                # PDF 파일 열기
                doc = fitz.open(file_path)
                page_count = len(doc)
                task_statuses[task_id]["page_count"] = page_count
                
                logs.append(f"PDF 로드 완료: {page_count}페이지")
                
                all_text = ""
                for page_num in range(page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    all_text += text + "\n\n"
                    logs.append(f"페이지 {page_num + 1} 처리 완료")
                
                doc.close()
                
                # 메타데이터 생성
                title_prompt = f"다음 문서의 제목을 한 줄로 요약해주세요:\n\n{all_text[:500]}"
                summary_prompt = f"다음 문서를 3-4문장으로 요약해주세요:\n\n{all_text[:1500]}"
                
                # GPT로 메타데이터 생성
                title_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": title_prompt}],
                    max_tokens=50
                )
                title = title_response.choices[0].message.content.strip()
                
                summary_response = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=200
                )
                summary = summary_response.choices[0].message.content.strip()
                
                response_style = "친절하고 상세한 설명을 제공하는 전문적인 톤"
                
                # 데이터베이스에 저장
                insert_stmt = documents.insert().values(
                    user_id=user_id,
                    pdf_name=filename,
                    page_number=1,
                    content=all_text,
                    title=title,
                    summary=summary,
                    response_style=response_style
                )
                result = await session.execute(insert_stmt)
                doc_id = result.inserted_primary_key[0]
                
                # 임베딩 생성 및 저장
                paragraphs = split_text_to_paragraphs(all_text)
                for paragraph in paragraphs:
                    embedding = await get_embedding_async(paragraph)
                    if embedding is not None:
                        emb_stmt = embeddings.insert().values(
                            user_id=user_id,
                            document_id=doc_id,
                            embedding=embedding.tobytes()
                        )
                        await session.execute(emb_stmt)
                
                await session.commit()
                logs.append(f"PDF 처리 완료: {filename}")
                
                # FAISS 인덱스 재로드
                await load_faiss_and_docstore()
                
                task_statuses[task_id]["status"] = "completed"
                
    except Exception as e:
        print(f"[ERROR] PDF 처리 실패: {e}")
        task_statuses[task_id]["status"] = "failed"
        logs.append(f"처리 실패: {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
