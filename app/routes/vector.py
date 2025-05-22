import unicodedata
import re
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.database import async_session, get_async_engine
from db.models import documents, embeddings
from app.config import settings
from app.vector_db import get_embedding  # 중복 제거: vector_db에서 임포트

router = APIRouter()

def safe_filename(name):
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name

def split_text_to_paragraphs(text):
    # 빈 줄(2개 이상의 개행) 또는 한 줄 개행 기준으로 문단 분리
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\r{2,}|\n|\r', text) if p.strip()]
    return paragraphs

upload_dir = Path("/tmp/temp_uploads")
upload_dir.mkdir(parents=True, exist_ok=True)

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    logs = []
    try:
        # 1. PDF 파일 저장
        filename = safe_filename(file.filename)
        file_path = upload_dir / f"temp_{filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())
        logs.append("PDF 파일이 서버에 저장됨.")

        # 2. DB에 PDF 원문 저장 및 문단 단위 임베딩
        doc = fitz.open(str(file_path))
        page_count = 0
        vector_count = 0
        async with async_session() as session:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    # 문서 저장
                    result = await session.execute(
                        insert(documents).values(
                            pdf_name=filename,
                            page_number=i,
                            content=text
                        ).returning(documents.c.id)
                    )
                    doc_id = result.scalar()
                    # 문단 단위로 분할
                    paragraphs = split_text_to_paragraphs(text)
                    for para in paragraphs:
                        if para.strip():
                            embedding = get_embedding(para)
                            await session.execute(
                                insert(embeddings).values(
                                    document_id=doc_id,
                                    embedding=embedding.tobytes()
                                )
                            )
                            vector_count += 1
                    page_count += 1
            await session.commit()
        logs.append("PDF가 DB에 저장됨.")
        logs.append("faiss 임베딩 벡터를 문단 단위로 분할하여 DB에 저장함.")

        os.remove(file_path)

        # 4. 전체 완료
        logs.append("전체 임베딩 및 메타데이터 DB 저장 완료.")

        return JSONResponse({
            "status": "success",
            "page_count": page_count,
            "vector_count": vector_count,
            "logs": logs
        })
    except Exception as e:
        logs.append(f"오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"status": "error", "logs": logs, "detail": str(e)}, status_code=500)