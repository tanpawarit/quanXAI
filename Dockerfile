# Dockerfile for quanXAI - using uv for fast package management
# Multi-stage build for optimized image size

# Stage 1: Build stage with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies (without dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Set PATH to use virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy source code and data
COPY src/ ./src/
COPY data/products_catalog.csv ./data/

# Create data directory for SQLite and Milvus Lite
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start API server
# Note: Run data ingestion first with: docker-compose run app python -m src.cli.ingest
CMD ["uvicorn", "src.application.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
