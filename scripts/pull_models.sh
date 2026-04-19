#!/usr/bin/env bash
# Idempotent: only pulls missing models.
set -euo pipefail

MODELS=(
  "qwen2.5vl:7b"
  "llama3.1:8b"
  "nomic-embed-text"
)

for m in "${MODELS[@]}"; do
  if ollama list | awk 'NR>1 {print $1}' | grep -Fxq "$m"; then
    echo "[skip] $m already present"
  else
    echo "[pull] $m"
    ollama pull "$m"
  fi
done

echo "All required models are available."
