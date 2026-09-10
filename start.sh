#!/usr/bin/env bash
set -e

PORT="${PORT:-8080}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WM_STORAGE_DIR="${WM_STORAGE_DIR:-/app/wm_storage}"

echo "=================================================================="
echo " Starting PyRete FARM Stack (FastAPI + Next.js Standalone)"
echo " -> Public Web Port:  ${PORT}"
echo " -> Internal Backend: ${BACKEND_PORT}"
echo " -> Persistence:      ${WM_STORAGE_DIR}"
echo "=================================================================="

export PYTHONPATH="/app/backend:${PYTHONPATH}"

# 1. Start FastAPI backend in background (0.0.0.0 so both internal Next.js and host port mapping can reach it)
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

# 2. Wait for FastAPI backend readiness using python standard library (no curl needed)
echo "⏳ [2/3] Waiting for FastAPI backend to initialize..."
MAX_WAIT=30
WAITED=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${BACKEND_PORT}/health')" > /dev/null 2>&1; do
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

# 3. Locate Next.js standalone server.js (ignoring any test files in node_modules)
echo "🚀 [3/3] Launching Next.js standalone server on port ${PORT}..."
SERVER_FILE=""
if [ -f "/app/frontend/server.js" ]; then
    SERVER_FILE="/app/frontend/server.js"
elif [ -f "/app/frontend/frontend/server.js" ]; then
    SERVER_FILE="/app/frontend/frontend/server.js"
else
    SERVER_FILE=$(find /app/frontend -not -path "*/node_modules/*" -name "server.js" | head -n 1)
fi

if [ -z "$SERVER_FILE" ] || [ ! -f "$SERVER_FILE" ]; then
    echo "❌ Error: Could not locate Next.js standalone server.js in /app/frontend"
    exit 1
fi

SERVER_DIR=$(dirname "$SERVER_FILE")
echo "📂 Located Next.js standalone server at: ${SERVER_FILE}"

# Ensure static assets are linked where server.js expects them
if [ -d "/app/frontend/.next/static" ] && [ ! -d "${SERVER_DIR}/.next/static" ]; then
    mkdir -p "${SERVER_DIR}/.next"
    cp -r /app/frontend/.next/static "${SERVER_DIR}/.next/static"
fi

cd "$SERVER_DIR"
echo "🌐 Starting Next.js: PORT=${PORT} HOSTNAME=0.0.0.0 node ${SERVER_FILE}"
PORT="${PORT}" HOSTNAME="0.0.0.0" node server.js &
FRONTEND_PID=$!

# Wait for either process to exit
wait -n "$BACKEND_PID" "$FRONTEND_PID"
EXIT_STATUS=$?
echo "⚠️ Process exited with status ${EXIT_STATUS}. Exiting PyRete..."
exit ${EXIT_STATUS}
