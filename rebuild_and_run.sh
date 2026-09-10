#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

IMAGE_NAME="pyrete-farm"
CONTAINER_NAME="pyrete-app"
FRONTEND_PORT="3000"
BACKEND_PORT="8000"

echo "=================================================================="
echo " 🔄 PyRete Rebuild & Persistent Container Deployment"
echo "=================================================================="

# 1. Verify Docker daemon is active
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker daemon is not running."
    echo "   Please open Docker Desktop on your Mac Mini and try again."
    exit 1
fi

# 2. Stop and remove any existing PyRete containers (by name or by image)
echo "🧹 [1/3] Stopping and removing existing PyRete containers..."
# Remove container with our target name if it exists
docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true

# Also remove any stray containers running the pyrete-farm image (e.g. epic_torvalds)
EXISTING_IDS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" -q)
if [ -n "$EXISTING_IDS" ]; then
    echo "   Cleaning up stray containers: ${EXISTING_IDS}"
    echo "$EXISTING_IDS" | xargs docker rm -f > /dev/null 2>&1 || true
fi

# Disable Buildx attestation manifests which cause Docker Desktop on Mac to hang at export
export DOCKER_BUILDX_NO_DEFAULT_ATTESTATIONS=1

# 3. Build Docker image fresh
echo "🔨 [2/3] Building Docker image '${IMAGE_NAME}' (multi-stage standalone)..."
docker build -t "${IMAGE_NAME}" .

# Print the new lean image size
echo ""
echo "📦 New image footprint:"
docker images "${IMAGE_NAME}" --format "   Size: {{.Size}} (Repository: {{.Repository}}:{{.Tag}})"

# 4. Create the named persistent container with port mappings (without starting it)
echo ""
echo "📦 [3/3] Creating persistent container '${CONTAINER_NAME}'..."
ENV_FLAG=""
if [ -n "$GEMINI_API_KEY" ]; then
    echo "🔑 Passing GEMINI_API_KEY from host environment..."
    ENV_FLAG="-e GEMINI_API_KEY=${GEMINI_API_KEY}"
fi

docker create \
    -p "${FRONTEND_PORT}:${FRONTEND_PORT}" \
    -p "${BACKEND_PORT}:${BACKEND_PORT}" \
    -e PORT="${FRONTEND_PORT}" \
    ${ENV_FLAG} \
    --name "${CONTAINER_NAME}" \
    "${IMAGE_NAME}" > /dev/null

echo ""
echo "=================================================================="
echo " ✅ Done building! Start the container with Docker Desktop."
echo "=================================================================="
echo "ℹ️  In Docker Desktop:"
echo "   1. Open Docker Desktop and go to the 'Containers' tab."
echo "   2. Locate '${CONTAINER_NAME}' in the list."
echo "   3. Click the Start (▶) button."
echo ""
echo "📍 Once started, your endpoints will be available at:"
echo "   - Frontend Dashboard: http://localhost:${FRONTEND_PORT}"
echo "   - Direct IP:          http://127.0.0.1:${FRONTEND_PORT}"
echo "   - FastAPI Docs:       http://localhost:${BACKEND_PORT}/docs"
echo "   - Health Check:       http://localhost:${FRONTEND_PORT}/health"
echo "=================================================================="
