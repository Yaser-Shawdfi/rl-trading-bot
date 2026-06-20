# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

COPY pyproject.toml ./
COPY rl_trader/ ./rl_trader/

RUN pip install --no-cache-dir build && python -m build

# ─── Runtime Stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Yaser Shawdfi <paiawon@outlook.com>"
LABEL description="Enterprise RL Trading Bot — PPO ensemble, FastAPI, Streamlit"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl fastapi uvicorn[standard] streamlit

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data models logs reports

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: start API
CMD ["uvicorn", "rl_trader.api:app", "--host", "0.0.0.0", "--port", "8000"]