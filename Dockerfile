## Dockerfile   
# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt mlflow

# COPY . .

# EXPOSE 8000
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

# System deps for nginx
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (FastAPI: api.py, Streamlit: app_streamlit.py, nginx.conf, start.sh)
COPY . .

# Nginx config: route "/" -> Streamlit(8502), "/api" -> FastAPI(8000)
COPY nginx.conf /etc/nginx/nginx.conf

# Expose single port 80 (nginx)
EXPOSE 80

# Make start script executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]



