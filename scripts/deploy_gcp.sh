#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy_gcp.sh  —  Manual deploy to GCP Cloud Run
# Usage: ./scripts/deploy_gcp.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID in .env or environment}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-doc-intel-api}"
REPO="doc-intel"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

# ── Ensure Artifact Registry repo exists ─────────────────────────────────────
echo "▸ Creating Artifact Registry repository (idempotent)..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null || true

# ── Configure Docker auth ─────────────────────────────────────────────────────
echo "▸ Configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ── Build + push ──────────────────────────────────────────────────────────────
echo "▸ Building image..."
docker build -t "${IMAGE}:latest" .

echo "▸ Pushing image..."
docker push "${IMAGE}:latest"

# ── Deploy ────────────────────────────────────────────────────────────────────
echo "▸ Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}:latest" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=10 \
  --port=8000 \
  --timeout=300 \
  --set-env-vars="OLLAMA_BASE_URL=${OLLAMA_BASE_URL},CHROMA_PATH=/tmp/chroma_db"

echo ""
echo "✓ Deployed! Service URL:"
gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)"

echo ""
echo "⚠  Note: On Cloud Run, ChromaDB uses /tmp (ephemeral)."
echo "   For production, mount a Cloud Filestore NFS or use a managed vector DB."
