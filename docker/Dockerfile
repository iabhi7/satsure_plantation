# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgdal-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for parallel processing
RUN pip install --no-cache-dir \
    multiprocess \
    psutil

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/processed outputs/models outputs/visualizations outputs/checkpoints

# Add entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import ee; ee.Initialize()" || exit 1

# Add proper labels
LABEL maintainer="Your Name <your.email@example.com>"
LABEL version="1.0"
LABEL description="Plantation Detection using Multi-Source Satellite Imagery"

ENTRYPOINT ["/entrypoint.sh"]