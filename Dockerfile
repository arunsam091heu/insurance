# Dockerfile (FastAPI + nginx)
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
# ensure fastapi + uvicorn are in requirements.txt; if unsure, add this next line:
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir "uvicorn[standard]"
COPY . .

# App files
COPY api.py preprocess.py start.sh nginx.conf ./

# Normalize start.sh and make executable
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["/app/start.sh"]


