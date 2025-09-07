#!/usr/bin/env bash
set -e

# Optional: download model at runtime if MODEL_URL is supplied
if [ -n "$MODEL_URL" ] && [ ! -f /app/model.pkl ]; then
  echo "Downloading model from $MODEL_URL ..."
  curl -fsSL "$MODEL_URL" -o /app/model.pkl
fi

# Start FastAPI
uvicorn api:app --host 0.0.0.0 --port 8000 --proxy-headers &

# Start nginx
nginx -g "daemon off;"
