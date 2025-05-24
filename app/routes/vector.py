import os
import shutil
import uuid
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession
from db.database import get_db, async_session
from db.models import documents, embeddings
import fitz
import numpy as np
from app.vector_db import get_embedding_async
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter()

# 환경 변수에서 설정 가져오기 (Docker 호환)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "vector_index")

# 디렉토리 생성
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(VECTOR_INDEX_DIR).mkdir(parents=True, exist_ok=True)

# FAISS 인덱스 경로
VECTOR_INDEX_PATH = os.path.join(VECTOR_INDEX_DIR, "faiss.index")

# 전역 상태 관리
task_statuses: Dict[str, Dict[str, Any]] = {}

def get_upload_dir() -> str:
    """업로드 디렉토리 경로 반환"""
    return UPLOAD_DIR

def get_vector_index_path() -> str:
    """벡터 인덱스 파일 경로 반환"""
    return VECTOR_INDEX_PATH

async def save_upload_file(file: UploadFile) -> str:
    """업로드된 파일을 저장하고 파일 경로 반환"""
    # 안전한 파일명 생성
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # 중복 파일 처리
    counter = 1
    name, ext = os.path.splitext(filename)
    while os.path.exists(file_path):
        filename = f"{name}_{counter}{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        counter += 1
    
    # 파일 저장
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"파일 저장 중 오류가 발생했습니다: {str(e)}")

async def process_pdf(
    task_id: str, 
    file_path: str, 
    filename: str, 
    session: AsyncSession,
    logs: List[str]
) -> None:
    """PDF 파일을 처리하고 벡터 데이터베이스에 저장"""
    try:
        # PDF 파일 열기
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"Processing PDF: {filename}, Pages: {total_pages}")
        except Exception as e:
            error_msg = f"PDF 파일을 열 수 없습니다: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # 페이지 수 제한 검사
        if total_pages > 50:
            error_msg = "PDF 페이지 수가 너무 많습니다. 50페이지 이하로 제한됩니다."
            logger.warning(error_msg)
            task_statuses[task_id] = {
                "status": "failed", 
                "logs": logs,
                "detail": error_msg
            }
            return

        # 트랜잭션 시작
        async with session.begin():
            all_embeddings = []
            
            # 각 페이지 처리
            for i in range(total_pages):
                try:
                    page = doc.load_page(i)
                    text = page.get_text().strip()
                    
                    if not text:
                        logs.append(f"페이지 {i+1}: 텍스트 없음")
                        continue

                    # 문서 메타데이터 저장
                    doc_data = {
                        "pdf_name": filename,
                        "page_number": i + 1,
                        "content": text,
                        "created_at": datetime.utcnow()
                    }

                    # 데이터베이스에 문서 저장
                    result = await session.execute(
                        insert(documents).values(**doc_data).returning(documents.c.id)
                    )
                    doc_id = result.scalar()

                    # 임베딩 생성
                    try:
                        embedding = await get_embedding_async(text)
                        if embedding is not None:
                            # 임베딩 데이터베이스에 저장
                            await session.execute(
                                insert(embeddings).values(
                                    document_id=doc_id,
                                    embedding=embedding.tobytes(),
                                    created_at=datetime.utcnow()
                                )
                            )
                            all_embeddings.append(embedding)
                            logs.append(f"페이지 {i+1}/{total_pages} 처리 완료")
                        else:
                            logs.append(f"페이지 {i+1}: 임베딩 생성 실패")
                    except Exception as e:
                        logger.error(f"페이지 {i+1} 임베딩 오류: {str(e)}")
                        logs.append(f"페이지 {i+1} 임베딩 오류: {str(e)}")
                        continue

                except Exception as e:
                    logger.error(f"페이지 {i+1} 처리 중 오류: {str(e)}")
                    logs.append(f"페이지 {i+1} 처리 중 오류: {str(e)}")
                    continue

            # FAISS 인덱스 업데이트
            if all_embeddings:
                await update_faiss_index(all_embeddings, logs)

            # 작업 상태 업데이트
            task_statuses[task_id] = {
                "status": "completed",
                "logs": logs,
                "page_count": total_pages,
                "vector_count": len(all_embeddings)
            }

    except Exception as e:
        error_msg = f"PDF 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        task_statuses[task_id] = {
            "status": "failed",
            "logs": logs,
            "detail": error_msg
        }
    finally:
        if 'doc' in locals():
            doc.close()


async def update_faiss_index(embeddings: List[np.ndarray], logs: List[str]) -> None:
    """FAISS 인덱스를 업데이트합니다."""
    try:
        import faiss
        
        # 임베딩이 비어있는 경우 무시
        if not embeddings:
            return
            
        # 첫 번째 임베딩의 차원 확인
        dimension = len(embeddings[0])
        
        # FAISS 인덱스 로드 또는 생성
        if os.path.exists(VECTOR_INDEX_PATH):
            try:
                index = faiss.read_index(VECTOR_INDEX_PATH)
                logs.append(f"기존 FAISS 인덱스 로드 완료: {VECTOR_INDEX_PATH}")
            except Exception as e:
                logs.append(f"인덱스 로드 실패, 새로 생성: {str(e)}")
                index = faiss.IndexFlatL2(dimension)
        else:
            index = faiss.IndexFlatL2(dimension)
            logs.append(f"새 FAISS 인덱스 생성 (차원: {dimension})")
        
        # 임베딩을 numpy 배열로 변환
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # 인덱스에 추가
        index.add(embeddings_array)
        
        # 인덱스 저장
        faiss.write_index(index, VECTOR_INDEX_PATH)
        logs.append(f"FAISS 인덱스 업데이트 완료 (벡터 수: {index.ntotal})")
        
    except Exception as e:
        error_msg = f"FAISS 인덱스 업데이트 중 오류: {str(e)}"
        logger.error(error_msg)
        logs.append(error_msg)
        raise

@router.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None
):
    """PDF 파일을 업로드하고 백그라운드에서 처리합니다."""
    logs = []
    file_path = None
    task_id = str(uuid.uuid4())
    
    try:
        # 파일 저장
        file_path = await save_upload_file(file)
        logs.append(f"PDF 저장 완료: {file.filename}")
        
        # 백그라운드 작업으로 PDF 처리 시작
        async with async_session() as session:
            background_tasks.add_task(
                process_pdf,
                task_id=task_id,
                file_path=file_path,
                filename=file.filename,
                session=session,
                logs=logs
            )
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "logs": logs,
            "message": "PDF 처리가 백그라운드에서 시작되었습니다."
        })
        
    except HTTPException as e:
        # HTTPException은 이미 처리된 오류
        raise e
    except Exception as e:
        error_msg = f"PDF 업로드 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logs.append(error_msg)
        
        # 오류 발생 시 임시 파일 정리
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"오류 발생으로 인한 임시 파일 삭제: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"임시 파일 삭제 실패: {str(cleanup_error)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "logs": logs, 
                "detail": str(e)
            }
        )

@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """작업 상태를 조회합니다.
    
    Args:
        task_id (str): 조회할 작업의 고유 ID
        
    Returns:
        JSONResponse: 작업 상태 정보
    """
    try:
        # 작업 상태 조회
        status = task_statuses.get(task_id)
        
        if status is None:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Task not found",
                    "task_id": task_id
                }
            )
            
        # 로그가 너무 길 경우 자르기
        if "logs" in status and isinstance(status["logs"], list):
            max_logs = 100  # 최대 로그 라인 수
            if len(status["logs"]) > max_logs:
                status["logs"] = status["logs"][-max_logs:]
                status["logs_truncated"] = True
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            **status
        })
        
    except Exception as e:
        error_msg = f"작업 상태 조회 중 오류 발생: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(e)
            }
        )