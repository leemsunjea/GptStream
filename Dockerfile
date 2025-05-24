FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (필요한 의존성 포함)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 비-root 사용자 생성 및 권한 설정
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /tmp/gptstream \
    && chown -R appuser:appuser /tmp/gptstream \
    && chmod 755 /tmp/gptstream

# 데이터 디렉토리 환경 변수 설정
ENV DATA_DIR=/tmp/gptstream/data
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 8000

# 종속성 복사 및 설치 (의존성 캐시 최적화를 위해 소스 복사 전에 수행)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 소유권 변경 및 실행 권한 부여
RUN chown -R appuser:appuser /app

# 비-root 사용자로 전환
USER appuser

# 실행 전 필요한 디렉토리 생성
RUN mkdir -p $DATA_DIR/uploads $DATA_DIR/vector_index

# 실행 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "25"]