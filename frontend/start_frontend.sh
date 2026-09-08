#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    if command -v pnpm &> /dev/null; then
        pnpm install
    else
        npm install
    fi
fi

echo "🚀 Starting Next.js Frontend on port 3000..."
echo "📍 Dashboard: http://localhost:3000"
if command -v pnpm &> /dev/null; then
    exec pnpm dev
else
    exec npm run dev
fi
