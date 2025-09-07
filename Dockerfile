FROM python:3.10-slim

# Nginx + curl for quick checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (make sure your requirements has fastapi, uvicorn[standard], streamlit, pandas, joblib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files explicitly (avoid .dockerignore surprises)
COPY api.py preprocess.py app_streamlit.py start.sh nginx.conf ./

# ✅ COPY your artifacts so Uvicorn won’t crash on startup
COPY model.pkl fraud_oracle.csv ./

# Nginx config + start script hygiene
COPY nginx.conf /etc/nginx/nginx.conf
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

EXPOSE 80
CMD ["/app/start.sh"]
