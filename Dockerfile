# Multi-stage build for minimal runtime image
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# Default envs (override in runtime)
ENV PORT=8000 \
    HOST=0.0.0.0 \
    POLICY_ENV_ONLY=true

# Create non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Healthcheck (optional; kube will use probes)
# HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
