

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV + torchvision need a handful of system libs even for CPU use.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libjpeg62-turbo \
      libpng16-16 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# --- Builder stage — install deps into a venv ---------------------------
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements.txt

# --- Runtime stage -------------------------------------------------------
FROM base AS runtime

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application code only — models / outputs / datasets are mounted at runtime
COPY src/        ./src/
COPY api/        ./api/
COPY inference/  ./inference/
COPY training/   ./training/
COPY configs/    ./configs/
COPY readme.md   ./readme.md

RUN mkdir -p /app/models /app/outputs /app/datasets \
 && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
