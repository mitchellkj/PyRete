#!/usr/bin/env bash
set -e

PORT="${PORT:-7860}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WM_STORAGE_DIR="${WM_STORAGE_DIR:-/app/wm_storage}"

echo "=================================================================="
echo " Starting PyRete FARM Stack (FastAPI + Next.js)"
echo " -> Frontend Port: ${PORT}"
echo " -> Backend Port:  ${BACKEND_PORT}"
echo " -> Persistence:   ${WM_STORAGE_DIR}"
echo "=================================================================="

export PYTHONPATH="/app/backend:${PYTHONPATH}"

# 1. Start FastAPI backend in background
echo "🚀 [1/3] Launching FastAPI backend on port ${BACKEND_PORT}..."
cd /app/backend
python -m uvicorn main:app --host 0.0.0.0 --port "${BACKEND_PORT}" &
BACKEND_PID=$!

# Trap signals for graceful shutdown
cleanup() {
    echo "🛑 Shutting down PyRete services..."
    kill -TERM "$BACKEND_PID" 2>/dev/null || true
    if [ -n "$FRONTEND_PID" ]; then
        kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 2. Wait for FastAPI backend readiness
echo "⏳ [2/3] Waiting for FastAPI backend to initialize..."
MAX_WAIT=30
WAITED=0
until curl -s "http://127.0.0.1:${BACKEND_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "❌ Error: FastAPI backend process died during startup."
        echo "   Showing backend process output:"
        wait "$BACKEND_PID" || true
        exit 1
    fi

    WAITED=$((WAITED + 1))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "⚠️ Backend health check timed out after ${MAX_WAIT}s. Proceeding with frontend launch..."
        break
    fi
    sleep 1
done

if [ "$WAITED" -lt "$MAX_WAIT" ]; then
    echo "✅ Backend is healthy and ready to accept requests!"
fi

# 3. Start Next.js frontend in foreground
echo "🚀 [3/3] Launching Next.js frontend on port ${PORT}..."
cd /app/frontend

# Use local next binary from node_modules
./node_modules/.bin/next start -p "${PORT}" &
FRONTEND_PID=$!

# Wait for either process to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID"
