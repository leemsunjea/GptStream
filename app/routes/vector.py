from fastapi import APIRouter, File, UploadFile, BackgroundTasks, Header, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import insert, delete, select # delete, select 추가
from db.database import async_session
from db.models import documents, embeddings
from app.vector_db import get_embedding_async, split_text_to_paragraphs, doc_store, index, load_faiss_and_docstore # load_faiss_and_docstore 추가
from asyncio import Lock # Lock 추가
import traceback # traceback 추가
import uuid # uuid 추가
import os # os 추가
import fitz # fitz (PyMuPDF) 추가
from openai import OpenAI # OpenAI 클라이언트 추가
from app.config import settings # 설정값 로드
import asyncio # asyncio 임포트 추가

router = APIRouter()
upload_dir = "/tmp/temp_uploads"
task_statuses = {}
doc_store_lock = Lock()
index_lock = Lock()
client = OpenAI(api_key=settings.OPENAI_API_KEY) # OpenAI 클라이언트 초기화

async def generate_metadata(text_content: str, filename: str):
    """문서 내용과 파일명을 기반으로 메타데이터를 생성합니다."""
    try:
        # 제목 생성 개선: 의미있는 첫 줄 또는 파일명 사용
        title = filename # 기본값으로 파일명
        if text_content:
            lines = [line.strip() for line in text_content.split('\n') if line.strip()] # 빈 줄 제외
            if lines:
                # 페이지 번호나 간단한 구분자로 시작하는 경우 제외 시도
                candidate_title = lines[0]
                # 숫자만 있거나, '-'로 시작하거나, 너무 짧은 경우는 제외
                if not (candidate_title.isdigit() or candidate_title.startswith('-') or len(candidate_title) < 5):
                    title = candidate_title[:150] # 제목 길이 제한
                elif len(lines) > 1: # 첫 줄이 부적합하면 두번째 줄도 고려
                    candidate_title_2 = lines[1]
                    if not (candidate_title_2.isdigit() or candidate_title_2.startswith('-') or len(candidate_title_2) < 5):
                        title = candidate_title_2[:150]

        # 요약 생성 (문서의 앞부분 2000자 사용)
        summary_prompt = f"""다음은 '{title}' 문서의 내용 일부입니다. 이 내용을 한국어로 2-3문장으로 요약해주세요:
---
{text_content[:2000]}
---
요약:"""
        summary_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=250 # 요약 토큰 증가 (기존 150 또는 200에서 증가)
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
        detailed_error = traceback.format_exc()
        print(f"[ERROR] 메타데이터 생성 중 오류: {e}\n{detailed_error}")
        # 실패 시 기본값 개선
        return {
            "title": filename,
            "summary": f"'{filename}' 문서의 내용을 요약하는 중 오류가 발생했습니다.",
            "response_style": "기본적인 응답 스타일을 사용합니다."
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
    doc_text_content_for_metadata = "" 
    doc_pymupdf = None # doc_pymupdf 초기화

    logs.append(f"처리 시작: {filename} (사용자: {user_id})")
    task_statuses[task_id] = {"status": "processing", "logs": logs, "page_count": 0, "filename": filename}

    try:
        # PDF 열기 및 전체 텍스트 추출 (메타데이터 생성용)
        try:
            doc_pymupdf = fitz.open(file_path)
            page_count = doc_pymupdf.page_count
            task_statuses[task_id]["page_count"] = page_count
            
            # 처음 몇 페이지의 텍스트를 합쳐서 메타데이터 생성에 사용 (예: 최대 3페이지 또는 5000자)
            chars_for_metadata = 0
            max_chars_for_metadata = 5000 
            for page_num in range(min(page_count, 3)): # 최대 3페이지까지만 메타데이터용으로 읽음
                page = doc_pymupdf.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    doc_text_content_for_metadata += page_text + "\n\n" # 페이지 구분을 위해 두 번의 줄바꿈
                    chars_for_metadata += len(page_text)
                    if chars_for_metadata > max_chars_for_metadata:
                        break
            
            if not doc_text_content_for_metadata.strip():
                 logs.append(f"경고: '{filename}'에서 메타데이터 생성을 위한 텍스트를 추출하지 못했습니다. 파일명 기반으로 메타데이터가 생성됩니다.")
                 doc_text_content_for_metadata = filename # 텍스트 추출 실패 시 파일명을 기본 내용으로 사용
        except Exception as e_parse:
            detailed_error_parse = traceback.format_exc()
            logs.append(f"PDF 파싱 오류 ({filename}): {e_parse}. 메타데이터는 파일명 기반으로 생성됩니다.\n{detailed_error_parse}")
            doc_text_content_for_metadata = filename # 파싱 오류 시에도 파일명을 기본 내용으로
            # 여기서 doc_pymupdf를 닫지 않고, 페이지 수가 0인 경우 등의 처리를 위해 아래로 이동
            # raise # 파싱 오류 시 처리를 중단하고 오류를 전파할 수 있도록 수정 -> 일단 메타데이터라도 생성 시도

        # 문서 전체에 대한 메타데이터 생성 (한 번만)
        metadata_dict = await generate_metadata(doc_text_content_for_metadata, filename)
        logs.append(f"메타데이터 생성됨: {metadata_dict} (사용자: {user_id}, 파일: {filename})")

        if page_count == 0: # PyMuPDF로 페이지 수를 얻었음에도 0인 경우 (매우 드묾)
            if doc_pymupdf and doc_pymupdf.page_count > 0:
                page_count = doc_pymupdf.page_count
                task_statuses[task_id]["page_count"] = page_count
                logs.append(f"페이지 수 재확인: {page_count} 페이지")
            else:
                logs.append(f"경고: '{filename}'의 페이지 수가 0이거나 PDF를 열 수 없습니다. 처리를 건너<0xEB><0x8F><0x84>니다.")
                # doc_pymupdf가 None일 수도 있으므로 확인 후 close
                if doc_pymupdf: doc_pymupdf.close()
                raise ValueError(f"'{filename}'에서 텍스트를 추출할 수 없고 페이지 수도 0입니다.")

        async with session_factory() as session:
            async with session.begin():
                for page_num in range(page_count):
                    current_page_obj = doc_pymupdf.load_page(page_num)
                    current_page_content = current_page_obj.get_text("text")
                    if not current_page_content.strip():
                        logs.append(f"페이지 {page_num + 1} 내용이 비어있어 건너<0xEB><0x8F><0x84>니다.")
                        continue

                    stmt_document = insert(documents).values(
                        user_id=user_id,
                        pdf_name=filename,
                        page_number=page_num + 1,
                        content=current_page_content,
                        title=metadata_dict["title"], 
                        summary=metadata_dict["summary"], 
                        response_style=metadata_dict["response_style"]
                    ).returning(documents.c.id) # returning 추가하여 id 바로 가져오기
                    result = await session.execute(stmt_document)
                    doc_id = result.scalar_one() # scalar_one() 또는 first()[0] 사용
                    logs.append(f"페이지 {page_num + 1} DB 저장됨 (doc_id: {doc_id}), 사용자: {user_id}")

                    paragraphs = split_text_to_paragraphs(current_page_content)
                    para_embeddings_count = 0
                    for para_text in paragraphs:
                        if not para_text.strip(): continue
                        embedding_vector = await get_embedding_async(para_text)
                        if embedding_vector is not None:
                            stmt_embedding = insert(embeddings).values(
                                document_id=doc_id,
                                user_id=user_id, 
                                embedding=embedding_vector.tobytes()
                            )
                            await session.execute(stmt_embedding)
                            para_embeddings_count += 1
                        else:
                            logs.append(f"페이지 {page_num + 1}의 문단에 대한 임베딩 생성 실패: '{para_text[:50]}...'")
                    logs.append(f"페이지 {page_num + 1}에 대해 {para_embeddings_count}개 문단 임베딩 생성 및 저장 완료, 사용자: {user_id}")
            # session.begin() 컨텍스트 매니저가 성공 시 자동 커밋
            
        logs.append(f"'{filename}' DB 처리 완료 (사용자: {user_id}). FAISS/doc_store 업데이트 시작...")
        
        async with doc_store_lock, index_lock:
             await load_faiss_and_docstore() # 이 함수는 자체 세션을 생성하여 사용
        
        logs.append(f"FAISS 인덱스 및 문서 저장소 업데이트 완료 (사용자: {user_id}, 파일: {filename})")
        processed_successfully = True

    except Exception as e:
        detailed_error = traceback.format_exc()
        error_message = f"PDF 처리 실패: ({type(e).__name__}) {e}\nSQL: {getattr(e, 'statement', 'N/A')}\nParams: {getattr(e, 'params', 'N/A')}\nTraceback: {detailed_error}"
        logs.append(error_message)
        print(f"[ERROR] process_pdf (task: {task_id}, user: {user_id}, file: {filename}): {error_message}")
        task_statuses[task_id]["status"] = "failed"
        task_statuses[task_id]["detail"] = f"({type(e).__name__}) {e}" 
    finally:
        if doc_pymupdf: # doc_pymupdf가 열렸었다면 반드시 닫아줌
            doc_pymupdf.close()
            logs.append(f"PDF 파일 핸들 닫힘: {filename}")

        if processed_successfully:
            task_statuses[task_id]["status"] = "completed"
            logs.append(f"'{filename}' 처리 성공적으로 완료 (사용자: {user_id})")
        else:
            if task_statuses[task_id].get("status") != "failed": # 이미 실패 상태가 아니면
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
    """
    특정 사용자의 모든 문서, 임베딩 데이터를 삭제하고,
    관련 FAISS 인덱스 및 인메모리 doc_store를 새로고침합니다.
    """
    logs = []
    logs.append(f"사용자 [{x_user_id}] 데이터 초기화 시작...")
    print(f"[INFO] 사용자 {x_user_id}의 데이터 초기화 요청 수신.")

    async with async_session() as session:
        async with session.begin(): # 트랜잭션 시작
            try:
                # 1. 해당 사용자의 문서 ID 조회
                doc_ids_result = await session.execute(
                    select(documents.c.id).where(documents.c.user_id == x_user_id)
                )
                doc_ids = [row[0] for row in doc_ids_result.fetchall()]
                logs.append(f"사용자 [{x_user_id}]의 문서 ID {len(doc_ids)}개 조회 완료: {doc_ids}")
                print(f"[DEBUG] 사용자 {x_user_id}의 문서 ID: {doc_ids}")

                if doc_ids:
                    # 2. 해당 문서 ID에 연결된 임베딩 삭제
                    delete_embeddings_stmt = delete(embeddings).where(embeddings.c.document_id.in_(doc_ids))
                    embedding_delete_result = await session.execute(delete_embeddings_stmt)
                    logs.append(f"사용자 [{x_user_id}]의 임베딩 {embedding_delete_result.rowcount}개 삭제 완료.")
                    print(f"[INFO] 사용자 {x_user_id}의 임베딩 {embedding_delete_result.rowcount}건 삭제 완료.")

                    # 3. 해당 사용자의 문서 삭제
                    delete_documents_stmt = delete(documents).where(documents.c.id.in_(doc_ids)) # user_id로 직접 삭제도 가능
                    document_delete_result = await session.execute(delete_documents_stmt)
                    logs.append(f"사용자 [{x_user_id}]의 문서 {document_delete_result.rowcount}개 삭제 완료.")
                    print(f"[INFO] 사용자 {x_user_id}의 문서 {document_delete_result.rowcount}건 삭제 완료.")
                else:
                    logs.append(f"사용자 [{x_user_id}]에게 삭제할 문서 데이터가 없습니다.")
                    print(f"[INFO] 사용자 {x_user_id}에게 삭제할 문서 데이터가 없습니다.")

                # 4. FAISS 인덱스 및 doc_store 재로드 (전역 상태 업데이트)
                logs.append(f"FAISS 인덱스 및 문서 저장소 재로드 시작...")
                print(f"[INFO] FAISS 인덱스 및 문서 저장소 재로드 시작 (사용자 {x_user_id} 데이터 초기화 후).")
                async with doc_store_lock, index_lock: # 전역 Lock 사용 가정
                    await load_faiss_and_docstore()
                logs.append(f"FAISS 인덱스 및 문서 저장소 재로드 완료.")
                print(f"[INFO] FAISS 인덱스 및 문서 저장소 재로드 완료.")
                
                await session.commit() # 명시적 커밋
                return JSONResponse({
                    "success": True, 
                    "message": f"사용자 [{x_user_id}]의 데이터가 성공적으로 초기화되었습니다.",
                    "logs": logs
                })

            except Exception as e:
                await session.rollback() # 명시적 롤백
                error_tb = traceback.format_exc()
                logs.append(f"데이터 초기화 중 서버 오류 발생: {str(e)}")
                print(f"[ERROR] 사용자 {x_user_id} 데이터 초기화 중 오류: {e}\n{error_tb}")
                # HTTPException 대신 JSONResponse 사용 통일성을 위해
                return JSONResponse(
                    status_code=500, 
                    content={"success": False, "detail": f"데이터 초기화 중 서버 오류 발생: {str(e)}", "logs": logs}
                )