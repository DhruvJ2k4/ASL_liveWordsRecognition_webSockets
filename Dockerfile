# Dockerfile (Python 3.7 + TensorFlow + OpenCV — Cloud Safe)

FROM python:3.7-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Your entrypoint (corrected path!)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
