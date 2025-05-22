from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import insert
from db.database import async_session
from db.models import documents, embeddings
import os
import fitz
import numpy as np
from app.vector_db import get_embedding_async, split_text_to_paragraphs
import asyncio
import uuid

router = APIRouter()
upload_dir = "/tmp/temp_uploads"
BATCH_SIZE = 5
task_statuses = {}  # 임시 상태 저장

async def save_upload_file(file: UploadFile, upload_dir: str):
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return file_path

async def process_pdf(task_id: str, file_path: str, filename: str, session, logs):
    task_statuses[task_id] = {"status": "pending", "logs": logs}
    try:
        doc = fitz.open(file_path)
        if len(doc) > 50:
            logs.append("PDF 페이지 수가 너무 많습니다. 50페이지 이하로 제한됩니다.")
            task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": "페이지 수 초과"}
            return
        async with session.begin():
            for i, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    logs.append(f"페이지 {i+1}: 텍스트 없음")
                    continue
                result = await session.execute(
                    insert(documents).values(
                        pdf_name=filename,
                        page_number=i,
                        content=text
                    ).returning(documents.c.id)
                )
                doc_id = result.scalar()
                logs.append(f"페이지 {i+1}: 문서 ID {doc_id} 저장")
                paragraphs = split_text_to_paragraphs(text)
                tasks = [get_embedding_async(para) for para in paragraphs]
                embeddings = await asyncio.gather(*tasks, return_exceptions=True)
                batch = []
                for j, emb in enumerate(embeddings):
                    if isinstance(emb, np.ndarray):
                        batch.append({"document_id": doc_id, "embedding": emb.tobytes()})
                if batch:
                    await session.execute(insert(embeddings), batch)
                    await session.commit()
                    logs.append(f"페이지 {i+1}: {len(batch)} 문단 임베딩 저장")
        logs.append("모든 페이지 처리 완료")
        task_statuses[task_id] = {"status": "completed", "logs": logs, "page_count": len(doc)}
    except Exception as e:
        logs.append(f"PDF 처리 오류: {e}")
        task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": str(e)}

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logs = []
    file_path = None
    task_id = str(uuid.uuid4())
    try:
        file_path = await save_upload_file(file, upload_dir)
        logs.append(f"PDF 저장 완료: {file.filename}")
        background_tasks.add_task(process_pdf, task_id, file_path, file.filename, async_session(), logs)
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "logs": logs,
            "message": "PDF 처리가 백그라운드에서 시작되었습니다."
        })
    except Exception as e:
        logs.append(f"전체 처리 오류: {e}")
        return JSONResponse({"success": False, "logs": logs, "detail": str(e)}, status_code=500)
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    status = task_statuses.get(task_id, {"status": "pending"})
    return JSONResponse(status)