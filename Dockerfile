# ─────────────────────────────────────────────────────────────────
#  PipeGuard AI — Dockerfile
#  Multi-stage build: keeps final image lean (~1.5 GB with torch)
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Build dependencies ──────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create non-root user (security best practice)
RUN useradd -m -u 1000 pipeguard

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY backend/   ./backend/
COPY frontend/  ./frontend/

# Create runtime directories with correct ownership
RUN mkdir -p models data logs && chown -R pipeguard:pipeguard /app

# Switch to non-root user
USER pipeguard

WORKDIR /app/backend

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
