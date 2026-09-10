#!/usr/bin/env bash
set -e

# ==============================================================================
# PyRete Cloud Run Deployment Script
# Tailored from heavy enterprise template to lightweight, public FARM stack
# ==============================================================================

TAG="${1:-latest}"

# Auto-detect or default GCP Project & Region
ACTIVE_GCP_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
PROJECT="${PROJECT:-${ACTIVE_GCP_PROJECT:-veytel-cloud-store}}"
REGION="${REGION:-us-west1}"

REPO="pyrete"
SERVICE_NAME="pyrete-app"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE_NAME}:${TAG}"

# Ensure we run from repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$ROOT_DIR")" = "scripts" ]; then
    ROOT_DIR="$(cd "$ROOT_DIR/.." && pwd)"
fi
cd "$ROOT_DIR"

echo "=================================================================="
echo " 🚀 Deploying PyRete FARM Stack to Google Cloud Run"
echo "=================================================================="
echo "📍 Directory:    $(pwd)"
echo "📍 GCP Project:  ${PROJECT}"
echo "📍 Region:       ${REGION}"
echo "📍 Service Name: ${SERVICE_NAME}"
echo "📍 Image URI:    ${IMAGE}"
echo "=================================================================="

# 1. Verify Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker daemon is not running."
    echo "   Please open Docker Desktop and try again."
    exit 1
fi

# 2. Verify gcloud authentication
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)
if [ -z "$CURRENT_ACCOUNT" ]; then
    echo "❌ ERROR: No active gcloud authentication found."
    echo "   Run 'gcloud auth login' and try again."
    exit 1
fi
echo "👤 Authenticated as: ${CURRENT_ACCOUNT}"

# 3. Configure Docker credential helper for Artifact Registry
echo "🔐 Configuring Docker authentication for ${REGION}-docker.pkg.dev..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet >/dev/null 2>&1

# 4. Create Artifact Registry repository if it doesn't already exist
echo "🗂 Checking for Artifact Registry repository: ${REPO} in ${REGION}..."
if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1; then
    echo "🗂 Creating Artifact Registry repository: ${REPO}..."
    gcloud artifacts repositories create "${REPO}" \
        --repository-format=docker \
        --location="${REGION}" \
        --project="${PROJECT}" \
        --description="PyRete container images repository"
    echo "✅ Artifact Registry repository created."
else
    echo "✅ Artifact Registry repository exists."
fi

# 5. Build and Push linux/amd64 Docker image
# Direct buildx push avoids heavy layer unpacking on macOS and handles multi-arch
echo "📦 Building & Pushing Docker image for linux/amd64..."
export DOCKER_BUILDX_NO_DEFAULT_ATTESTATIONS=1

docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --sbom=false \
    -t "${IMAGE}" \
    --push .

echo "✅ Image pushed successfully to ${IMAGE}"

# 6. Optional environment variables (e.g. Gemini API Key if present on host)
ENV_VARS_FLAG=""
if [ -n "$GEMINI_API_KEY" ]; then
    echo "🔑 Forwarding GEMINI_API_KEY from host environment..."
    ENV_VARS_FLAG="--set-env-vars=GEMINI_API_KEY=${GEMINI_API_KEY}"
fi

# 7. Deploy to Google Cloud Run
# Tailored configuration:
# - memory=1Gi, cpu=1: comfortably hosts both FastAPI + Next.js standalone
# - min-instances=0: scales to 0 when idle ($0 cost when not in use)
# - max-instances=2: safety cap against accidental runaway traffic
# - allow-unauthenticated: public access for portfolio/demo
# - NO VPC, NO custom egress, NO JWT Secret Manager required
echo "🚀 Deploying service to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --project="${PROJECT}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=1Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=2 \
    ${ENV_VARS_FLAG} \
    --quiet

# 8. Retrieve public Service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT}" --region="${REGION}" --format='value(status.url)')

echo ""
echo "=================================================================="
echo " 🎉 PyRete Cloud Run Deployment Successful!"
echo "=================================================================="
echo "🌐 Public URL:   ${SERVICE_URL}"
echo "📍 Health Check: curl ${SERVICE_URL}/health"
echo "📍 API Docs:     ${SERVICE_URL}/api/docs (proxied via Next.js)"
echo "=================================================================="
