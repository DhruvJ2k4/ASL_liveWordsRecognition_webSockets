# Dockerfile
FROM python:3.7-slim

# Needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose backend port
EXPOSE 8000

# Start WebSocket server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
