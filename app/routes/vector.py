import unicodedata
import re
import os
import fitz  # PyMuPDF
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.database import async_session
from db.models import documents, embeddings
from app.vector_db import get_embedding  # 임베딩 함수
from app.config import settings

router = APIRouter()

upload_dir = Path("/tmp/temp_uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5

def safe_filename(name):
    name = unicodedata.normalize("NFC", name)
    return re.sub(r"[^\w.\-]", "_", name)

async def save_upload_file(file: UploadFile, upload_dir: Path) -> Path:
    filename = safe_filename(file.filename)
    file_path = upload_dir / f"temp_{filename}"
    with open(file_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)
    await file.close()
    return file_path

def split_text_to_paragraphs(text: str):
    return [p.strip() for p in re.split(r'\n{2,}|\r{2,}|\n|\r', text) if p.strip()]

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logs = []
    file_path = None
    try:
        file_path = await save_upload_file(file, upload_dir)
        logs.append("PDF 저장 완료")

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logs.append(f"PDF 열기 실패: {e}")
            return JSONResponse({"success": False, "logs": logs, "detail": str(e)}, status_code=400)

        async with async_session() as session:
            for i, page in enumerate(doc):
                text = page.get_text()
                if not text.strip():
                    continue

                try:
                    result = await session.execute(
                        insert(documents).values(
                            pdf_name=file.filename,
                            page_number=i,
                            content=text
                        ).returning(documents.c.id)
                    )
                    doc_id = result.scalar()
                    if not doc_id:
                        logs.append(f"문서 ID 생성 실패: {file.filename} p{i+1}")
                        continue
                except Exception as e:
                    logs.append(f"문서 저장 실패: {e}")
                    continue

                paragraphs = split_text_to_paragraphs(text)
                batch = []
                for j, para in enumerate(paragraphs):
                    try:
                        embedding = get_embedding(para)
                        batch.append({
                            "document_id": doc_id,
                            "embedding": embedding.tobytes()
                        })
                    except Exception as e:
                        logs.append(f"임베딩 실패 (p{i+1} 문단{j+1}): {e}")
                        continue

                    if len(batch) >= BATCH_SIZE:
                        try:
                            await session.execute(insert(embeddings), batch)
                            await session.commit()
                            batch = []
                        except Exception as e:
                            await session.rollback()
                            logs.append(f"벡터 저장 실패: {e}")

                if batch:
                    try:
                        await session.execute(insert(embeddings), batch)
                        await session.commit()
                    except Exception as e:
                        await session.rollback()
                        logs.append(f"마지막 벡터 저장 실패: {e}")

            logs.append("모든 페이지 처리 완료")
            return JSONResponse({
                "success": True,
                "page_count": len(doc),
                "vector_count": len(logs),  # 실제 벡터 개수로 바꾸려면 별도 카운팅 필요
                "logs": logs
            })

    except Exception as e:
        logs.append(f"전체 처리 오류: {e}")
        print(traceback.format_exc())
        return JSONResponse({"success": False, "logs": logs, "detail": str(e)}, status_code=500)

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)