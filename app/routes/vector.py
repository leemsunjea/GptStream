import unicodedata
import re
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import fitz  # PyMuPDF
import openai
import faiss
import numpy as np
from app.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.database import async_session, get_async_engine
from db.models import documents, embeddings

router = APIRouter()

openai.api_key = settings.OPENAI_API_KEY

dimension = 1536  # OpenAI Embedding 차원
index = None
doc_store = []

def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype='float32')

def safe_filename(name):
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name

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

        # 2. DB에 PDF 원문 저장
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
                    # 임베딩 저장
                    embedding = get_embedding(text)
                    await session.execute(
                        insert(embeddings).values(
                            document_id=doc_id,
                            embedding=embedding.tobytes()
                        )
                    )
                    page_count += 1
                    vector_count += 1
        logs.append("PDF가 DB에 저장됨.")
        logs.append("faiss 임베딩 벡터를 메타값 단위로 분할하여 DB에 저장함.")

        await session.commit()
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