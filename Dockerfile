# =============================================================================
# Stage 1: Build Next.js Production Frontend (Standalone Mode)
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies using standard npm (built into node:20-slim)
COPY frontend/package.json ./
RUN npm install

# Copy frontend source and compile standalone Next.js server
COPY frontend ./
RUN npm run build

# =============================================================================
# Stage 2: Ultra-Lean Production Runner (Python 3.11 + Minimal Node Runtime)
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# 1. Install runtime dependencies (libstdc++6 is required by Node.js binary) and git for pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy standalone Node.js binary from frontend-builder (~90MB, no npm, no apt)
COPY --from=frontend-builder /usr/local/bin/node /usr/local/bin/node

# Verify Node.js runtime works
RUN node -v

# 3. Copy backend application source
COPY backend /app/backend

# 4. Copy pre-persisted working memory cache (49 pre-computed queries)
COPY wm_storage /app/wm_storage

# 5. Copy Next.js standalone server and static assets (minimal pruned modules)
COPY --from=frontend-builder /app/frontend/.next/standalone /app/frontend
COPY --from=frontend-builder /app/frontend/.next/static /app/frontend/.next/static

# 6. Copy and prepare production start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Environment variables
# Cloud Run injects PORT (default 8080), local override via -e PORT=3000
ENV PYTHONPATH=/app/backend \
    PORT=8080 \
    BACKEND_PORT=8000 \
    BACKEND_URL=http://127.0.0.1:8000 \
    WM_STORAGE_DIR=/app/wm_storage \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production \
    HOSTNAME="0.0.0.0"

# Expose ports:
# - 8080: Next.js Frontend (Cloud Run default ingress)
# - 8000: Internal FastAPI Backend
EXPOSE 8080
EXPOSE 8000

# Launch services
CMD ["/app/start.sh"]
