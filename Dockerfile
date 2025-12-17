FROM python:3.11-slim

# Install system dependencies for sentencepiece
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid issues with cached packages
RUN pip install --no-cache-dir -r requirements.txt

# Verify sentencepiece installation
RUN python -c "import sentencepiece; print(f'SentencePiece version: {sentencepiece.__version__}')"

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads results/checkpoints src/data

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "60"]
