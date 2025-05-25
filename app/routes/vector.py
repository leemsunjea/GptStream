from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from sqlalchemy import insert
from db.database import async_session
from db.models import documents, embeddings
import os
import fitz
import numpy as np
from app.vector_db import get_embedding_async, split_text_to_paragraphs, doc_store, index, load_faiss_and_docstore
import asyncio
import uuid
from asyncio import Lock

router = APIRouter()
upload_dir = "/tmp/temp_uploads"
task_statuses = {}
doc_store_lock = Lock()
index_lock = Lock()

async def save_upload_file(file: UploadFile, upload_dir: str):
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"[DEBUG] 파일 저장 성공: {file_path}")
    except Exception as e:
        print(f"[DEBUG] 파일 저장 실패: {e}")
        raise
    return file_path

async def process_pdf(task_id: str, file_path: str, filename: str, session, logs, user_id: str): # user_id 추가
    async with doc_store_lock, index_lock:
        logs.append(f"처리할 파일 경로: {file_path} (사용자: {user_id})") # 로그에 user_id 추가
        try:
            doc = fitz.open(file_path)
            if len(doc) > 50:
                logs.append("PDF 페이지 수가 너무 많습니다. 50페이지 이하로 제한됩니다.")
                task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": "페이지 수 초과"}
                return
            async with session.begin():
                # all_embeddings_for_user_pdf = [] # 현재 PDF 처리에서 생성된 임베딩만 저장 (load_faiss_and_docstore로 이전)
                # current_pdf_doc_store_references = [] # 현재 PDF의 텍스트와 메타데이터 임시 저장 (load_faiss_and_docstore로 이전)

                for i, page in enumerate(doc):
                    text = page.get_text()
                    if not text.strip():
                        logs.append(f"페이지 {i+1}: 텍스트 없음")
                        print(f"[DEBUG] 페이지 {i+1}: 텍스트 없음 (사용자: {user_id})")
                        continue

                    result = await session.execute(
                        insert(documents).values(
                            user_id=user_id,  # user_id 추가
                            pdf_name=filename,
                            page_number=i,
                            content=text
                        ).returning(documents.c.id)
                    )
                    doc_id = result.scalar()
                    logs.append(f"페이지 {i+1}: 문서 ID {doc_id} 저장 (사용자: {user_id})")
                    print(f"[DEBUG] 저장된 문서 ID: {doc_id}, 페이지 번호: {i+1}, 사용자: {user_id}, 내용: {text[:30]}...", flush=True)
                    await asyncio.sleep(0)
                    paragraphs = split_text_to_paragraphs(text)
                    tasks = [get_embedding_async(para) for para in paragraphs]
                    embedding_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    batch_embeddings_for_db = []
                    for j, emb_result in enumerate(embedding_results):
                        if isinstance(emb_result, Exception):
                            logs.append(f"임베딩 오류 (페이지 {i+1}, 문단 {j+1}): {emb_result} (사용자: {user_id})")
                            continue
                        if isinstance(emb_result, np.ndarray):
                            batch_embeddings_for_db.append({"document_id": doc_id, "embedding": emb_result.tobytes(), "user_id": user_id}) # user_id 추가
                            # all_embeddings_for_user_pdf.append(emb_result) # FAISS 인덱스용 (load_faiss_and_docstore로 이전)
                            # current_pdf_doc_store_references.append({
                            #     'text': paragraphs[j],
                            #     'user_id': user_id,
                            #     'doc_id': doc_id
                            # }) # (load_faiss_and_docstore로 이전)
                        else:
                            logs.append(f"잘못된 임베딩 결과 유형 (페이지 {i+1}, 문단 {j+1}): {type(emb_result)} (사용자: {user_id})")

                    if batch_embeddings_for_db:
                        stmt = insert(embeddings)
                        await session.execute(stmt, batch_embeddings_for_db)
                        logs.append(f"페이지 {i+1}: {len(batch_embeddings_for_db)} 문단 임베딩 저장 (사용자: {user_id})")
                
                # PDF DB 저장 완료 후, 전체 FAISS 인덱스 및 문서 저장소 재로드
                await load_faiss_and_docstore() # DB에서 전체 데이터를 다시 로드하여 인덱스와 doc_store 갱신
                logs.append(f"FAISS 인덱스 및 문서 저장소 업데이트 완료 (사용자: {user_id})")

                logs.append(f"모든 페이지 처리 완료 (사용자: {user_id})")
                task_statuses[task_id] = {"status": "completed", "logs": logs, "page_count": len(doc)}
        except Exception as e:
            logs.append(f"PDF 처리 오류 (사용자: {user_id}): {e}")
            task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": str(e)}
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[DEBUG] 임시 파일 삭제: {file_path} (사용자: {user_id})")
            except Exception as e:
                print(f"[ERROR] 임시 파일 삭제 실패: {e} (사용자: {user_id})")

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, x_user_id: str = Header(..., description="클라이언트 UUID")): # x_user_id 추가
    logs = []
    file_path = None
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status": "pending", "logs": logs} # 초기 상태 설정
    try:
        file_path = await save_upload_file(file, upload_dir)
        logs.append(f"PDF 저장 완료: {file.filename} (사용자: {x_user_id})")
        # process_pdf 호출 시 x_user_id 전달
        background_tasks.add_task(process_pdf, task_id, file_path, file.filename, async_session(), logs, x_user_id)
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "logs": logs,
            "message": f"PDF 처리가 백그라운드에서 시작되었습니다. (사용자: {x_user_id})"
        })
    except Exception as e:
        logs.append(f"전체 처리 오류 (사용자: {x_user_id}): {e}")
        # 실패 시 task_statuses 업데이트
        task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": str(e)}
        return JSONResponse({"success": False, "logs": logs, "detail": str(e)}, status_code=500)

@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    status = task_statuses.get(task_id, {"status": "pending"})
    return JSONResponse(status)