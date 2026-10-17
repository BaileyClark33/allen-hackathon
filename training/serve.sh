#!/usr/bin/env bash
# Serve the merged fine-tuned router with vLLM on the Nebius AI Cloud VM.
# The OpenAI-compatible endpoint is http://<vm-ip>:8000/v1/ ; put that in config.yaml under
# models.nano_ft.base_url and the same key in .env as ROUTER_FT_API_KEY.
#
#   ROUTER_FT_API_KEY=... bash training/serve.sh runs/router-ft/merged
set -euo pipefail
MODEL_DIR="${1:-runs/router-ft/merged}"
: "${ROUTER_FT_API_KEY:?set ROUTER_FT_API_KEY}"
exec vllm serve "$MODEL_DIR" \
  --served-model-name nemoroute-router-ft \
  --api-key "$ROUTER_FT_API_KEY" \
  --trust-remote-code \
  --max-model-len 4096 \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  ${VLLM_EXTRA_ARGS:-}
