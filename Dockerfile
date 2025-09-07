# Dockerfile
FROM python:3.10-slim

# System deps for nginx (and certs)
RUN apt-get update && apt-get install -y --no-install-recommends \
      nginx ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Python deps ----------
COPY requirements.txt .
# Make sure requirements.txt includes: fastapi uvicorn[standard] streamlit pandas joblib
RUN pip install --no-cache-dir -r requirements.txt

# ---------- App code ----------
# (copy files explicitly to avoid .dockerignore surprises)
COPY api.py preprocess.py app_streamlit.py start.sh nginx.conf ./
# Model/data needed at startup (so Uvicorn won't crash)
COPY model.pkl fraud_oracle.csv ./

# Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Normalize Windows CRLF -> LF and ensure start.sh is executable
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

EXPOSE 80
CMD ["/app/start.sh"]
