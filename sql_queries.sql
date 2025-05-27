-- 생성된 메타데이터를 확인하는 SQL 명령어들

-- 1. 모든 문서의 메타데이터 조회 (최신순)
SELECT 
    id,
    user_id,
    pdf_name,
    page_number,
    title,
    summary,
    response_style,
    created_at,
    SUBSTRING(content, 1, 100) || '...' AS content_preview
FROM documents 
ORDER BY created_at DESC;

-- 2. 특정 사용자의 문서 메타데이터 조회
SELECT 
    id,
    pdf_name,
    page_number,
    title,
    summary,
    response_style,
    created_at
FROM documents 
WHERE user_id = 'YOUR_USER_ID_HERE'
ORDER BY created_at DESC;

-- 3. PDF 파일별 메타데이터 그룹화
SELECT 
    pdf_name,
    COUNT(*) as page_count,
    MIN(created_at) as uploaded_at,
    MAX(title) as document_title,
    MAX(summary) as document_summary,
    MAX(response_style) as document_response_style
FROM documents 
GROUP BY pdf_name, user_id
ORDER BY uploaded_at DESC;

-- 4. 최근 업로드된 문서 5개의 상세 정보
SELECT 
    d.id,
    d.user_id,
    d.pdf_name,
    d.title,
    d.summary,
    d.response_style,
    d.created_at,
    COUNT(e.id) as embedding_count
FROM documents d
LEFT JOIN embeddings e ON d.id = e.document_id
GROUP BY d.id, d.user_id, d.pdf_name, d.title, d.summary, d.response_style, d.created_at
ORDER BY d.created_at DESC
LIMIT 5;

-- 5. 문서별 임베딩 개수 확인
SELECT 
    d.pdf_name,
    d.page_number,
    d.title,
    COUNT(e.id) as embedding_count
FROM documents d
LEFT JOIN embeddings e ON d.id = e.document_id
GROUP BY d.id, d.pdf_name, d.page_number, d.title
ORDER BY d.pdf_name, d.page_number;

-- 6. 사용자별 문서 통계
SELECT 
    user_id,
    COUNT(DISTINCT pdf_name) as pdf_count,
    COUNT(*) as total_pages,
    COUNT(DISTINCT title) as unique_titles,
    MIN(created_at) as first_upload,
    MAX(created_at) as last_upload
FROM documents
GROUP BY user_id;

-- 7. 특정 PDF의 모든 페이지 메타데이터
SELECT 
    page_number,
    title,
    summary,
    response_style,
    SUBSTRING(content, 1, 200) || '...' AS content_preview
FROM documents 
WHERE pdf_name = 'YOUR_PDF_NAME_HERE'
ORDER BY page_number;

-- 8. 제목이 설정된 문서만 조회
SELECT 
    pdf_name,
    page_number,
    title,
    summary,
    created_at
FROM documents 
WHERE title IS NOT NULL AND title != ''
ORDER BY created_at DESC;

-- 9. 응답 스타일별 문서 분류
SELECT 
    response_style,
    COUNT(*) as document_count,
    COUNT(DISTINCT pdf_name) as pdf_count
FROM documents 
WHERE response_style IS NOT NULL
GROUP BY response_style;

-- 10. 메타데이터가 누락된 문서 찾기
SELECT 
    id,
    pdf_name,
    page_number,
    title,
    summary,
    response_style
FROM documents 
WHERE title IS NULL 
   OR summary IS NULL 
   OR response_style IS NULL;


DELETE FROM embeddings;
DELETE FROM documents;