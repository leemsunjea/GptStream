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
        response_style_prompt = f"""이 문서는 '{title}'에 관한 내용이며, 주요 내용은 다음과 같습니다: '{summary}'. 
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
            f.write(await file.read())
        print(f"[DEBUG] 파일 저장 성공: {file_path}")
    except Exception as e:
        print(f"[DEBUG] 파일 저장 실패: {e}")
        raise
    return file_path

async def process_pdf(task_id: str, file_path: str, filename: str, session_factory, logs, user_id: str): # session -> session_factory
    processed_successfully = False
    # doc_store_lock과 index_lock은 load_faiss_and_docstore 호출까지 포함하도록 유지합니다.
    # process_pdf 함수 자체는 백그라운드 태스크로 동시에 여러개가 실행될 수 있지만,
    # 공유 자원인 doc_store와 index에 대한 접근 및 수정은 동기화되어야 합니다.
    # load_faiss_and_docstore 내부에서 DB를 읽고 전역 변수를 업데이트하므로,
    # 이 함수 호출 자체를 lock으로 감싸는 것이 더 안전할 수 있습니다.
    # 하지만 현재 lock은 process_pdf 시작부터 걸려있으므로,
    # 한 번에 하나의 PDF 처리 + load_faiss_and_docstore만 실행됩니다.
    # 만약 PDF 처리(DB 저장)는 병렬로 하고, load_faiss_and_docstore만 동기화하려면 lock 위치 조정 필요.
    # 현재 구조에서는 process_pdf 전체를 lock으로 감싸는 것이 가장 간단하고 안전합니다.
    async with doc_store_lock, index_lock:
        logs.append(f"처리할 파일 경로: {file_path} (사용자: {user_id})")
        doc_object_for_page_count = None # 페이지 수 참조를 위해 try 블록 외부에서 선언
        first_page_text_for_metadata = ""
        try:
            async with session_factory() as session: # 새로운 세션 사용
                async with session.begin(): # 트랜잭션 시작
                    doc_object_for_page_count = fitz.open(file_path) # fitz.open 결과를 변수에 저장
                    if len(doc_object_for_page_count) > 0:
                        # 첫 페이지 내용으로 메타데이터 생성 시도
                        first_page_text_for_metadata = doc_object_for_page_count[0].get_text() if doc_object_for_page_count[0] else ""
                    
                    # 메타데이터 생성
                    metadata_dict = await generate_metadata(first_page_text_for_metadata, filename)
                    logs.append(f"메타데이터 생성 완료: {metadata_dict} (사용자: {user_id})")

                    if len(doc_object_for_page_count) > 50:
                        logs.append("PDF 페이지 수가 너무 많습니다. 50페이지 이하로 제한됩니다.")
                        task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": "페이지 수 초과"}
                        # 임시 파일 삭제 로직 추가
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"[DEBUG] 임시 파일 삭제 (페이지 수 초과): {file_path} (사용자: {user_id})")
                        except Exception as e_remove:
                            print(f"[ERROR] 임시 파일 삭제 실패 (페이지 수 초과): {e_remove} (사용자: {user_id})")
                        return # 여기서 함수 종료

                    for i, page in enumerate(doc_object_for_page_count): # 저장된 변수 사용
                        text = page.get_text()
                        if not text.strip():
                            logs.append(f"페이지 {i+1}: 텍스트 없음")
                            print(f"[DEBUG] 페이지 {i+1}: 텍스트 없음 (사용자: {user_id})")
                            continue

                        # 메타데이터를 포함하여 문서 정보 저장
                        result = await session.execute(
                            insert(documents).values(
                                user_id=user_id,
                                pdf_name=filename,
                                page_number=i,
                                content=text,
                                title=metadata_dict.get("title"),
                                summary=metadata_dict.get("summary"),
                                response_style=metadata_dict.get("response_style")
                            ).returning(documents.c.id)
                        )
                        doc_id = result.scalar()
                        logs.append(f"페이지 {i+1}: 문서 ID {doc_id} 저장 (메타데이터 포함) (사용자: {user_id})")
                        print(f"[DEBUG] 저장된 문서 ID: {doc_id}, 페이지 번호: {i+1}, 사용자: {user_id}, 내용: {text[:30]}...", flush=True)
                        
                        paragraphs = split_text_to_paragraphs(text)
                        tasks = [get_embedding_async(para) for para in paragraphs]
                        embedding_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        batch_embeddings_for_db = []
                        for j, emb_result in enumerate(embedding_results):
                            if isinstance(emb_result, Exception):
                                logs.append(f"임베딩 오류 (페이지 {i+1}, 문단 {j+1}): {emb_result} (사용자: {user_id})")
                                continue
                            if isinstance(emb_result, np.ndarray):
                                batch_embeddings_for_db.append({"document_id": doc_id, "embedding": emb_result.tobytes(), "user_id": user_id})
                            else:
                                logs.append(f"잘못된 임베딩 결과 유형 (페이지 {i+1}, 문단 {j+1}): {type(emb_result)} (사용자: {user_id})")

                        if batch_embeddings_for_db:
                            stmt = insert(embeddings)
                            await session.execute(stmt, batch_embeddings_for_db)
                            logs.append(f"페이지 {i+1}: {len(batch_embeddings_for_db)} 문단 임베딩 저장 (사용자: {user_id})")
                    
                    # session.begin() 블록이 여기서 끝나면 커밋됨
                processed_successfully = True 
                logs.append(f"모든 페이지 DB 저장 및 커밋 완료 (사용자: {user_id})")

            # DB 작업이 커밋된 후에 FAISS 인덱스 및 문서 저장소 재로드
            if processed_successfully:
                await load_faiss_and_docstore() # DB에서 전체 데이터를 다시 로드하여 인덱스와 doc_store 갱신
                logs.append(f"FAISS 인덱스 및 문서 저장소 업데이트 완료 (사용자: {user_id})")
                # doc_object_for_page_count가 None이 아닐 때만 len 호출
                page_count_to_log = len(doc_object_for_page_count) if doc_object_for_page_count else 0
                task_statuses[task_id] = {"status": "completed", "logs": logs, "page_count": page_count_to_log}
            else:
                if task_statuses.get(task_id, {}).get("status") != "failed":
                     task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": "PDF 처리 중 DB 작업 실패 또는 처리되지 않음"}

        except Exception as e:
            logs.append(f"PDF 처리 오류 (사용자: {user_id}): {e}")
            task_statuses[task_id] = {"status": "failed", "logs": logs, "detail": str(e)}
        finally:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[DEBUG] 임시 파일 삭제: {file_path} (사용자: {user_id})")
            except Exception as e_remove:
                print(f"[ERROR] 임시 파일 삭제 실패: {e_remove} (사용자: {user_id})")

@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, x_user_id: str = Header(..., description="클라이언트 UUID")):
    logs = []
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