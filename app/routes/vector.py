import os
import json
import uuid
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any, Type, Callable
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form, Depends, status
from fastapi.responses import JSONResponse
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader, BaseLoader
)
from app.config import settings
from db.database import get_async_session, get_user_data_dir, get_or_create_session_id

# 로깅 설정
logger = logging.getLogger(__name__)

# 라우터 설정
router = APIRouter()

# 허용되는 파일 확장자와 해당 로더 매핑
ALLOWED_EXTENSIONS: Dict[str, Type[BaseLoader]] = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.docx': Docx2txtLoader,
    '.pptx': UnstructuredPowerPointLoader,
}

def get_file_loader(file_path: str) -> Optional[Type[BaseLoader]]:
    """파일 확장자에 따라 적절한 로더 반환"""
    ext = os.path.splitext(file_path)[1].lower()
    return ALLOWED_EXTENSIONS.get(ext)

def ensure_directory(path: str) -> None:
    """디렉토리가 존재하는지 확인하고 없으면 생성"""
    try:
        os.makedirs(path, exist_ok=True, mode=0o777)
        os.chmod(path, 0o777)
    except Exception as e:
        logger.error(f"디렉토리 생성 실패 {path}: {e}")
        raise

def get_user_upload_dir(session_id: str) -> str:
    """사용자별 업로드 디렉토리 경로 반환"""
    user_dir = get_user_data_dir(session_id)
    upload_dir = os.path.join(user_dir, "uploads")
    ensure_directory(upload_dir)
    return upload_dir

def get_user_vector_dir(session_id: str) -> str:
    """사용자별 벡터 인덱스 디렉토리 경로 반환"""
    user_dir = get_user_data_dir(session_id)
    vector_dir = os.path.join(user_dir, "vector_index")
    ensure_directory(vector_dir)
    return vector_dir

# 텍스트 분할기 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# 세션별 FAISS 인덱스와 문서 저장소를 저장할 딕셔너리
session_stores: Dict[str, Dict[str, Any]] = {}  # session_id -> {"faiss_index": ..., "docstore": ...}

# 임베딩 모델 차원 (사용하는 모델에 따라 조정 필요)
EMBEDDING_DIM = 1536  # 예: OpenAI text-embedding-ada-002

# 임베딩 생성 함수 (실제 구현에서는 외부 API 또는 로컬 모델 호출)
async def get_embedding(text: str) -> List[float]:
    """텍스트를 벡터로 변환 (임시 구현)"""
    # 실제로는 여기서 OpenAI API나 다른 임베딩 모델을 호출합니다.
    # 예시로 간단하게 해시 기반 벡터 생성
    import hashlib
    import numpy as np
    
    # 텍스트의 해시값을 기반으로 고정 길이 벡터 생성
    hash_obj = hashlib.sha256(text.encode())
    hash_int = int(hash_obj.hexdigest(), 16) % (10**8)
    
    # 간단한 벡터 생성 (실제로는 모델을 사용해야 함)
    np.random.seed(hash_int)
    return list(np.random.rand(EMBEDDING_DIM).astype(np.float32))

def load_or_create_index(session_id: str) -> Dict[str, Any]:
    """세션 ID에 해당하는 FAISS 인덱스와 문서 저장소를 로드하거나 생성"""
    if session_id not in session_stores:
        vector_dir = get_user_vector_dir(session_id)
        index_path = os.path.join(vector_dir, "index.faiss")
        docstore_path = os.path.join(vector_dir, "docstore.json")
        
        try:
            # 기존 인덱스가 있으면 로드
            if os.path.exists(index_path) and os.path.exists(docstore_path):
                index = faiss.read_index(index_path)
                with open(docstore_path, 'r', encoding='utf-8') as f:
                    docstore = json.load(f)
            else:
                # 새 인덱스 생성
                index = faiss.IndexFlatL2(EMBEDDING_DIM)
                docstore = []
                
        except Exception as e:
            logger.error(f"인덱스 로드 실패: {e}")
            # 오류 발생 시 새 인덱스 생성
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            docstore = []
        
        session_stores[session_id] = {
            "faiss_index": index,
            "docstore": docstore
        }
    
    return session_stores[session_id]

def save_index(session_id: str) -> None:
    """세션의 FAISS 인덱스와 문서 저장소를 파일에 저장"""
    if session_id in session_stores:
        try:
            vector_dir = get_user_vector_dir(session_id)
            os.makedirs(vector_dir, exist_ok=True, mode=0o777)
            
            index_path = os.path.join(vector_dir, "index.faiss")
            docstore_path = os.path.join(vector_dir, "docstore.json")
            
            # FAISS 인덱스 저장
            faiss.write_index(session_stores[session_id]["faiss_index"], index_path)
            
            # 문서 저장소 저장
            with open(docstore_path, 'w', encoding='utf-8') as f:
                json.dump(session_stores[session_id]["docstore"], f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {e}")

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """파일을 업로드하고 벡터 인덱스에 추가"""
    try:
        # 파일 확장자 확인
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원하지 않는 파일 형식입니다: {file_ext}"
            )
        
        # 업로드 디렉토리 확인 및 생성
        upload_dir = get_user_upload_dir(session_id)
        os.makedirs(upload_dir, exist_ok=True, mode=0o777)
        
        # 파일 저장 경로
        file_path = os.path.join(upload_dir, file.filename)
        
        # 파일 저장 (덮어쓰기 방지)
        if os.path.exists(file_path):
            base, ext = os.path.splitext(file.filename)
            counter = 1
            while os.path.exists(os.path.join(upload_dir, f"{base}_{counter}{ext}")):
                counter += 1
            file_path = os.path.join(upload_dir, f"{base}_{counter}{ext}")
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 파일 로드 및 처리
        loader_class = get_file_loader(file_path)
        if not loader_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"파일을 처리할 수 없습니다: {file.filename}"
            )
        
        # 문서 로드 및 분할
        loader = loader_class(file_path)
        documents = loader.load()
        
        # 텍스트 분할
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc.page_content))
        
        # 벡터 인덱스 로드 또는 생성
        store = load_or_create_index(session_id)
        
        # 각 텍스트 청크에 대해 임베딩 생성 및 인덱스에 추가
        for text in texts:
            # 임베딩 생성 (실제로는 비동기로 처리하는 것이 좋음)
            embedding = await get_embedding(text)
            
            # FAISS 인덱스에 추가
            vector = np.array([embedding], dtype=np.float32)
            store["faiss_index"].add(vector)
            
            # 문서 저장소에 메타데이터 저장
            doc_id = str(uuid.uuid4())
            store["docstore"].append({
                "id": doc_id,
                "text": text,
                "source": file.filename,
                "vector": embedding
            })
        
        # 인덱스 저장
        save_index(session_id)
        
        return {"status": "success", "message": f"파일이 성공적으로 처리되었습니다. {len(texts)}개의 청크가 추가되었습니다."}
        
    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/search")
async def search_similar(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(5),
    session_id: str = Form(...)
):
    """벡터 유사도 검색"""
    try:
        # 벡터 인덱스 로드
        store = load_or_create_index(session_id)
        
        # 쿼리 임베딩 생성
        query_embedding = await get_embedding(query)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # 유사도 검색
        distances, indices = store["faiss_index"].search(query_vector, k=min(top_k, len(store["docstore"])))
        
        # 결과 조회
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(store["docstore"]):
                doc = store["docstore"][idx]
                results.append({
                    "text": doc["text"],
                    "source": doc.get("source", "unknown"),
                    "score": float(distance)
                })
        
        return {"query": query, "results": results}
        
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        )
os.makedirs(VECTOR_INDEX_BASE_DIR, exist_ok=True, mode=0o777)

# 의존성 주입을 위한 헬퍼 함수
async def get_user_session(request: Request):
    """요청에서 세션 ID를 가져와 DB 세션 반환"""
    session_id = get_or_create_session_id(request)
    session_factory = await get_async_session(session_id)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_session_store(session_id: str) -> dict:
    """세션 ID에 해당하는 저장소를 가져오거나 생성합니다."""
    if session_id not in session_stores:
        # 세션별 벡터 인덱스 디렉토리 생성
        session_vector_dir = os.path.join(VECTOR_INDEX_BASE_DIR, session_id)
        os.makedirs(session_vector_dir, exist_ok=True, mode=0x1ff)  # 0o777
        
        # 세션 저장소 초기화
        session_stores[session_id] = {
            "faiss_index": None,
            "docstore": {},
            "index_to_doc_id": {},
            "doc_id_to_index": {},
            "vector_dir": session_vector_dir
        }
        
        # 기존 인덱스 로드 시도
        index_path = os.path.join(session_vector_dir, "faiss.index")
        docstore_path = os.path.join(session_vector_dir, "docstore.pkl")
        
        try:
            # FAISS 인덱스 로드
            if os.path.exists(index_path):
                session_stores[session_id]["faiss_index"] = faiss.read_index(index_path)
                logger.info(f"✅ 세션 {session_id} FAISS 인덱스 로드 완료")
            
            # 문서 저장소 로드
            if os.path.exists(docstore_path):
                with open(docstore_path, "rb") as f:
                    session_stores[session_id]["docstore"] = pickle.load(f)
                
                # 매핑 테이블 생성
                docstore = session_stores[session_id]["docstore"]
                session_stores[session_id]["index_to_doc_id"] = {
                    i: doc_id for i, doc_id in enumerate(docstore.keys())
                }
                session_stores[session_id]["doc_id_to_index"] = {
                    doc_id: i for i, doc_id in session_stores[session_id]["index_to_doc_id"].items()
                }
                logger.info(f"✅ 세션 {session_id} 문서 저장소 로드 완료: {len(docstore)}개 문서")
                
        except Exception as e:
            logger.error(f"세션 {session_id} 인덱스 로드 오류: {e}")
            # 오류 발생 시 초기화
            session_stores[session_id] = {
                "faiss_index": None,
                "docstore": {},
                "index_to_doc_id": {},
                "doc_id_to_index": {},
                "vector_dir": session_vector_dir
            }
    
    return session_stores[session_id]

async def load_faiss_and_docstore():
    """애플리케이션 시작 시 호출되어 기본 세션 저장소를 초기화합니다."""
    # 기본 세션 저장소 초기화
    get_session_store("default")
    logger.info("✅ 기본 벡터 저장소 초기화 완료")

async def save_faiss_and_docstore(session_id: str = None):
    """지정된 세션의 FAISS 인덱스와 문서 저장소를 저장합니다.
    
    Args:
        session_id: 저장할 세션 ID. None이면 모든 세션 저장
    """
    if session_id:
        # 특정 세션만 저장
        if session_id in session_stores:
            await _save_session_store(session_id)
    else:
        # 모든 세션 저장
        for sid in list(session_stores.keys()):
            await _save_session_store(sid)

async def _save_session_store(session_id: str):
    """특정 세션의 저장소를 저장합니다."""
    store = session_stores.get(session_id)
    if not store or store["faiss_index"] is None:
        logger.warning(f"세션 {session_id}: 저장할 FAISS 인덱스가 없습니다.")
        return
    
    try:
        # 디렉토리 생성
        os.makedirs(store["vector_dir"], exist_ok=True, mode=0x1ff)  # 0o777
        
        # FAISS 인덱스 저장
        index_path = os.path.join(store["vector_dir"], "faiss.index")
        faiss.write_index(store["faiss_index"], index_path)
        
        # 문서 저장소 저장
        docstore_path = os.path.join(store["vector_dir"], "docstore.pkl")
        with open(docstore_path, "wb") as f:
            pickle.dump(store["docstore"], f)
            
        logger.info(f"✅ 세션 {session_id}: FAISS 인덱스와 문서 저장소 저장 완료")
        
    except Exception as e:
        logger.error(f"세션 {session_id}: 저장 중 오류 발생 - {str(e)}")
        raise

async def save_upload_file(request: Request, file: UploadFile) -> str:
    """업로드된 파일을 저장하고 파일 경로 반환"""
    # 안전한 파일명 생성
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    
    # 세션 ID 기반 디렉토리 경로 가져오기
    session_id = get_or_create_session_id(request)
    upload_dir = get_user_upload_dir(session_id)
    vector_dir = get_user_vector_dir(session_id)
    
    # 디렉토리 생성
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(vector_dir, exist_ok=True)
    
    # 파일 저장 경로 생성
    file_path = os.path.join(upload_dir, filename)
    
    # 중복 파일 처리
    counter = 1
    name, ext = os.path.splitext(filename)
    while os.path.exists(file_path):
        filename = f"{name}_{counter}{ext}"
        file_path = os.path.join(upload_dir, filename)
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
    logs: List[str],
    request: Request
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
                await update_faiss_index(all_embeddings, logs, request)

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

async def update_faiss_index(embeddings: List[np.ndarray], logs: List[str], request: Request) -> None:
    """FAISS 인덱스를 업데이트합니다."""
    try:
        import faiss
        import numpy as np
        
        if not embeddings:
            logs.append("업데이트할 임베딩이 없습니다.")
            return
            
        # 세션 ID 가져오기
        session_id = get_or_create_session_id(request)
        session_store = await get_session_store(session_id)
        
        # 첫 번째 임베딩의 차원 확인
        dimension = len(embeddings[0])
        
        # FAISS 인덱스 로드 또는 생성
        if session_store["faiss_index"] is None:
            logs.append("새 FAISS 인덱스 생성")
            session_store["faiss_index"] = faiss.IndexFlatL2(dimension)
        
        # 임베딩을 numpy 배열로 변환
        embeddings_array = np.array(embeddings, dtype='float32')
        
        # FAISS 인덱스에 벡터 추가
        session_store["faiss_index"].add(embeddings_array)
        
        # 문서 저장소 업데이트 (예시로 임시 문서 추가)
        # 실제로는 여기서 문서 내용을 적절히 처리해야 함
        for _ in range(len(embeddings)):
            session_store["docstore"].append("새 문서 내용")  # 실제 문서 내용으로 대체 필요
        
        # 매핑 테이블 업데이트
        docstore = session_store["docstore"]
        session_store["index_to_doc_id"] = {
            i: doc_id for i, doc_id in enumerate(docstore.keys())
        }
        session_store["doc_id_to_index"] = {
            doc_id: i for i, doc_id in session_store["index_to_doc_id"].items()
        }
        
        # 세션 저장소 저장
        await save_faiss_and_docstore(session_id)
        logs.append(f"FAISS 인덱스 업데이트 완료 (벡터 수: {session_store['faiss_index'].ntotal})")
        
    except Exception as e:
        error_msg = f"FAISS 인덱스 업데이트 중 오류 발생: {str(e)}"
        logs.append(error_msg)
        logger.error(error_msg)
        raise

@router.post("/upload_pdf")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_user_session)
):
    """PDF 파일을 업로드하고 백그라운드에서 처리합니다."""
    logs = []
    file_path = None
    task_id = str(uuid.uuid4())
    
    try:
        # 파일 저장
        file_path = await save_upload_file(request, file)
        logs.append(f"PDF 저장 완료: {file.filename}")
        
        # 백그라운드 작업으로 PDF 처리 시작
        async with async_session() as session:
            background_tasks.add_task(
                process_pdf,
                task_id=task_id,
                file_path=file_path,
                filename=file.filename,
                session=session,
                logs=logs,
                request=request
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