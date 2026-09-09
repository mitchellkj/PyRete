# =============================================================================
# Stage 1: Build Next.js Production Frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install pnpm package manager
RUN npm install -g pnpm

# Install dependencies using lockfile for reproducible builds
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy frontend source code and compile Next.js production bundle
COPY frontend ./
RUN pnpm build

# =============================================================================
# Stage 2: Final Runtime Image (Python 3.11 + Node.js 20)
# =============================================================================
FROM python:3.11-slim

# Install system dependencies:
# - git: Required for pip to clone py_rete repository from GitHub
# - curl: Used for downloading Node.js and performing health checks
# - ca-certificates: Secure SSL connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend application source
COPY backend /app/backend

# Copy pre-persisted working memory cache (49 pre-computed product x currency queries)
COPY wm_storage /app/wm_storage

# Copy compiled Next.js frontend (including node_modules and .next build artifacts)
COPY --from=frontend-builder /app/frontend /app/frontend

# Copy and prepare entrypoint script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Configure Hugging Face Spaces non-root user (UID 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

# Set runtime environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/app/backend \
    PORT=7860 \
    BACKEND_PORT=8000 \
    BACKEND_URL=http://127.0.0.1:8000 \
    WM_STORAGE_DIR=/app/wm_storage \
    PYTHONUNBUFFERED=1

# Expose ports:
# - 7860: Next.js Frontend (Hugging Face Spaces default public port)
# - 8000: FastAPI Backend (API endpoints & Swagger documentation)
EXPOSE 7860
EXPOSE 8000

# Launch both FastAPI and Next.js services
CMD ["/app/start.sh"]
