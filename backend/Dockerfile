# Build from the repository root: docker build -f backend/Dockerfile .
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app (keeps the `backend.` package path the code imports from)
COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
