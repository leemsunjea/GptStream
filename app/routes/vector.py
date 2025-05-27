from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import insert, delete, select # delete 추가
from db.database import async_session
from db.models import documents, embeddings
import os
import fitz  # PyMuPDF
import numpy as np
from app.vector_db import get_embedding_async, split_text_to_paragraphs, doc_store, index, load_faiss_and_docstore
import asyncio
import uuid
from asyncio import Lock
from openai import OpenAI # OpenAI 클라이언트 추가
from app.config import settings # 설정값 로드

router = APIRouter()
upload_dir = "/tmp/temp_uploads"
task_statuses = {}
doc_store_lock = Lock()
index_lock = Lock()
client = OpenAI(api_key=settings.OPENAI_API_KEY) # OpenAI 클라이언트 초기화

async def generate_metadata(text_content: str, filename: str):
    """문서 내용과 파일명을 기반으로 메타데이터를 생성합니다."""
    try:
        # 제목 생성 (첫 번째 줄 또는 파일명 활용)
        title = text_content.split('\\n')[0][:100] if text_content else filename

        # 요약 생성
        summary_prompt = f"""다음 텍스트를 한국어로 2-3문장으로 요약해주세요:
---
{text_content[:2000]} # API 길이 제한 고려
---
요약:"""
        summary_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150
            )
        )
        summary = summary_response.choices[0].message.content.strip()

        # 응답 스타일 제안
        response_style_prompt = f"""이 문서는 '{title}'에 관한 내용이며, 주요 내용은 '{summary}'. 
이 문서를 참고하여 사용자에게 답변할 때 어떤 스타일로 응답하는 것이 좋을지 한국어로 간단히 제안해주세요. (예: 친절하고 상세하게, 전문적이고 간결하게 등)
응답 스타일 제안:"""
        response_style_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": response_style_prompt}],
                max_tokens=50
            )
        )
        response_style = response_style_response.choices[0].message.content.strip()

        return {
            "title": title,
            "summary": summary,
            "response_style": response_style
        }
    except Exception as e:
        print(f"[ERROR] 메타데이터 생성 중 오류: {e}")
        return {
            "title": filename, # 오류 시 파일명을 기본 제목으로 사용
            "summary": "요약 생성 중 오류 발생",
            "response_style": "기본 응답 스타일"
        }

async def save_upload_file(file: UploadFile, upload_dir: str):
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read() # Read content from UploadFile
            f.write(content) # Write to file
        print(f"[DEBUG] 파일 저장 성공: {file_path}")
    except Exception as e:
        print(f"[DEBUG] 파일 저장 실패: {e}")
        raise
    return file_path

async def process_pdf(task_id: str, file_path: str, filename: str, session_factory, logs, user_id: str):
    processed_successfully = False
    page_count = 0
    
    logs.append(f"처리 시작: {filename} (사용자: {user_id})")
    task_statuses[task_id] = {"status": "processing", "logs": logs, "page_count": 0, "filename": filename}

    try:
        # Create a new session for this task using the passed session_factory
        async with session_factory() as session:
            async with session.begin(): # Start a transaction
                logs.append(f"DB 세션 시작됨 (사용자: {user_id}, 파일: {filename})")
                
                doc = fitz.open(file_path)
                page_count = len(doc)
                task_statuses[task_id]["page_count"] = page_count
                logs.append(f"'{filename}' 에서 {page_count} 페이지 로드됨 (사용자: {user_id})")

                first_page_text_for_metadata = ""
                if page_count > 0:
                    first_page_text_for_metadata = doc[0].get_text("text")
                else:
                    logs.append(f"'{filename}'에 페이지가 없어 메타데이터 생성을 건너뜁니다.")
                    # If no pages, we might still want to create a document entry with no content
                    # or handle as an error. For now, let's assume it might proceed with no pages.

                # Generate metadata using the content of the first page (or whole doc if preferred)
                metadata_dict = await generate_metadata(first_page_text_for_metadata, filename)
                logs.append(f"메타데이터 생성됨: {metadata_dict} (사용자: {user_id}, 파일: {filename})")

                if page_count == 0: # Handle case with no pages after metadata generation attempt
                    logs.append(f"'{filename}'에 처리할 페이지가 없습니다. DB 저장을 건너뜁니다.")
                
                for page_num in range(page_count):
                    page_content = doc[page_num].get_text("text")
                    if not page_content.strip():
                        logs.append(f"페이지 {page_num + 1} 내용이 비어있어 건너뜁니다 (사용자: {user_id}).")
                        continue

                    stmt_doc = insert(documents).values(
                        user_id=user_id,
                        pdf_name=filename,
                        page_number=page_num,
                        content=page_content,
                        title=metadata_dict["title"],
                        summary=metadata_dict["summary"],
                        response_style=metadata_dict["response_style"]
                        # created_at은 DB에서 자동으로 설정됨 (server_default=func.now())
                    ).returning(documents.c.id)
                    
                    result = await session.execute(stmt_doc)
                    doc_id = result.scalar_one()
                    logs.append(f"페이지 {page_num + 1} DB 저장됨 (doc_id: {doc_id}), 사용자: {user_id}")

                    paragraphs = split_text_to_paragraphs(page_content)
                    para_embeddings_count = 0
                    for para_text in paragraphs:
                        if not para_text.strip():
                            continue
                        embedding_vector = await get_embedding_async(para_text)
                        if embedding_vector is not None:
                            embedding_bytes = embedding_vector.tobytes()
                            stmt_emb = insert(embeddings).values(
                                user_id=user_id,
                                document_id=doc_id,
                                embedding=embedding_bytes
                            )
                            await session.execute(stmt_emb)
                            para_embeddings_count += 1
                    logs.append(f"페이지 {page_num + 1}에 대해 {para_embeddings_count}개 문단 임베딩 생성 및 저장 완료, 사용자: {user_id}")
            
            # session.begin() context manager handles commit on successful exit
            logs.append(f"'{filename}' DB 처리 완료 (사용자: {user_id}). FAISS/doc_store 업데이트 시작...")
            
            # Lock for FAISS index and doc_store updates
            async with doc_store_lock, index_lock:
                 await load_faiss_and_docstore() # This function creates its own session
            
            logs.append(f"FAISS 인덱스 및 문서 저장소 업데이트 완료 (사용자: {user_id}, 파일: {filename})")
            processed_successfully = True

    except Exception as e:
        import traceback
        detailed_error = traceback.format_exc()
        error_message = f"PDF 처리 실패: ({type(e).__name__}) {e}\nSQL: {getattr(e, 'statement', 'N/A')}\nParams: {getattr(e, 'params', 'N/A')}\nTraceback: {detailed_error}"
        logs.append(error_message)
        print(f"[ERROR] process_pdf (task: {task_id}, user: {user_id}, file: {filename}): {error_message}")
        task_statuses[task_id]["status"] = "failed"
        task_statuses[task_id]["detail"] = f"({type(e).__name__}) {e}" # Store a simpler error for client
    finally:
        if processed_successfully:
            task_statuses[task_id]["status"] = "completed"
            logs.append(f"'{filename}' 처리 성공적으로 완료 (사용자: {user_id})")
        else:
            if task_statuses[task_id].get("status") != "failed":
                task_statuses[task_id]["status"] = "failed"
                task_statuses[task_id]["detail"] = "알 수 없는 오류로 처리 실패"
            logs.append(f"'{filename}' 처리 중 문제 발생 (사용자: {user_id})")

        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logs.append(f"임시 파일 삭제됨: {file_path}")
            except Exception as e_remove:
                logs.append(f"임시 파일 삭제 실패: {file_path}, 오류: {e_remove}")
        
        task_statuses[task_id]["logs"] = logs
        print(f"[INFO] process_pdf 완료 (task: {task_id}, status: {task_statuses[task_id]['status']}, user: {user_id}, file: {filename})")

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, x_user_id: str = Header(..., description="클라이언트 UUID")):
    logs = [] # 각 업로드 요청에 대한 초기 로그 리스트
    file_path = None
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {"status": "pending", "logs": logs}
    try:
        file_path = await save_upload_file(file, upload_dir)
        logs.append(f"PDF 저장 완료: {file.filename} (사용자: {x_user_id})")
        # process_pdf 호출 시 async_session (세션 팩토리) 전달
        background_tasks.add_task(process_pdf, task_id, file_path, file.filename, async_session, logs, x_user_id) # async_session() -> async_session
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

@router.post("/reset_user_data")
async def reset_user_data(x_user_id: str = Header(..., description="클라이언트 UUID")):
    async with async_session() as session:
        async with session.begin():
            try:
                # 1. 해당 사용자의 임베딩 데이터 삭제
                # 먼저 삭제할 문서 ID들을 가져옵니다.
                stmt_get_doc_ids = select(documents.c.id).where(documents.c.user_id == x_user_id)
                result_doc_ids = await session.execute(stmt_get_doc_ids)
                doc_ids_to_delete = [row[0] for row in result_doc_ids.fetchall()]

                if doc_ids_to_delete:
                    stmt_delete_embeddings = delete(embeddings).where(embeddings.c.document_id.in_(doc_ids_to_delete))
                    await session.execute(stmt_delete_embeddings)
                    print(f"[INFO] 사용자 {x_user_id}의 임베딩 데이터 삭제 완료 (문서 ID: {doc_ids_to_delete})")
                else:
                    print(f"[INFO] 사용자 {x_user_id}에 대한 문서가 없어 임베딩 데이터 삭제를 건너뜁니다.")

                # 2. 해당 사용자의 문서 데이터 삭제
                stmt_delete_documents = delete(documents).where(documents.c.user_id == x_user_id)
                await session.execute(stmt_delete_documents)
                print(f"[INFO] 사용자 {x_user_id}의 문서 데이터 삭제 완료")
                
                # 3. FAISS 인덱스 및 doc_store 재로드 (전체 재로드)
                # 특정 사용자 데이터만 선택적으로 제거하는 것은 FAISS 인덱스 구조상 복잡할 수 있으므로,
                # 여기서는 전체 재로드를 통해 반영합니다.
                # 주의: 이 방식은 다른 사용자 데이터에도 영향을 줄 수 있으므로, 
                # 사용자별 격리가 중요하다면 FAISS 인덱스 관리 전략 수정 필요.
                async with doc_store_lock, index_lock:
                    await load_faiss_and_docstore()
                print(f"[INFO] FAISS 인덱스 및 문서 저장소 재로드 완료 (사용자 {x_user_id} 데이터 삭제 후)")

                await session.commit()
                return JSONResponse({"success": True, "message": "사용자 데이터가 성공적으로 초기화되었습니다."})
            except Exception as e:
                await session.rollback()
                print(f"[ERROR] 사용자 {x_user_id} 데이터 초기화 중 오류: {e}")
                raise HTTPException(status_code=500, detail=f"데이터 초기화 중 서버 오류 발생: {e}")