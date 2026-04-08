# Dockerfile — Code Debugging RL Environment
# Follows OpenEnv containerization spec for HF Spaces deployment

FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY server/ ./server/
COPY inference.py .

# HF Spaces listens on port 7860
EXPOSE 7860

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]