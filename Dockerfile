FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including git-lfs for model files
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads src/data results/checkpoints

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "60"]
