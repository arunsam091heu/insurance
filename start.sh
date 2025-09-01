#!/usr/bin/env bash
set -e

# Start Uvicorn (FastAPI) in background
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (frontend) in background; let API_URL default to '/api/predict-raw' behind nginx
export API_URL="${API_URL:-/api/predict-raw}"
streamlit run app_streamlit.py --server.port 8502 --server.address 0.0.0.0 &

# Start Nginx in foreground
nginx -g "daemon off;"
