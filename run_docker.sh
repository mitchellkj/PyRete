#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=================================================================="
echo " 🐳 Building and Running PyRete FARM Stack in Docker"
echo "=================================================================="

# Check if Docker daemon is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker daemon is not running."
    echo "   Please start Docker Desktop on your Mac Mini and try again."
    exit 1
fi

IMAGE_NAME="pyrete-farm"
CONTAINER_NAME="pyrete-app"

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing previously running container '${CONTAINER_NAME}'..."
    docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

# Disable Buildx attestation manifests which cause Docker Desktop on Mac to hang at export
export DOCKER_BUILDX_NO_DEFAULT_ATTESTATIONS=1

# Build Docker image
echo "🔨 Building Docker image '${IMAGE_NAME}' (this may take a few minutes on first run)..."
docker buildx build --provenance=false --sbom=false --load -t "${IMAGE_NAME}" .

echo ""
echo "=================================================================="
echo "🚀 Starting container '${CONTAINER_NAME}'..."
echo "📍 Frontend Dashboard: http://localhost:3000"
echo "📍 FastAPI Docs:       http://localhost:8000/docs"
echo "📍 Health Endpoint:    http://localhost:3000/health"
echo "=================================================================="
echo "ℹ️  Press Ctrl+C to stop the container."
echo ""

# Run container mapping ports 3000 (Frontend) and 8000 (Backend)
# Optional: pass GEMINI_API_KEY from host environment if available
ENV_FLAG=""
if [ -n "$GEMINI_API_KEY" ]; then
    echo "🔑 Passing GEMINI_API_KEY from host environment..."
    ENV_FLAG="-e GEMINI_API_KEY=${GEMINI_API_KEY}"
fi

exec docker run -it \
    -p 3000:3000 \
    -p 8000:8000 \
    ${ENV_FLAG} \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}"
