# Dockerfile voor Schoolrecht Backend (FastAPI / Python 3.11)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Systeempakketten
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies installeren
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Applicatiebroncode kopiëren
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "gpo_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
