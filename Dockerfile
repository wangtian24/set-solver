FROM python:3.11-slim

WORKDIR /app

# Install system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch)
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy application code and weights
COPY src/ src/
COPY weights/ weights/

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "7860"]
