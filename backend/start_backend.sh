#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV_PYTHON="../venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

echo "🚀 Starting PyRete FastAPI Backend on port 8000..."
echo "📍 API Docs: http://localhost:8000/docs"
exec "$VENV_PYTHON" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
