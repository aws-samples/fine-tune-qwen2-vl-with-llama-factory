# Use NVIDIA CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/conda/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh


# Create a new conda environment and install dependencies
RUN conda create -n qwen_env python=3.10 -y
SHELL ["conda", "run", "-n", "qwen_env", "/bin/bash", "-c"]

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Copy your application code
WORKDIR /app
COPY app.py .
COPY start.sh .

# Make the start script executable
RUN chmod +x start.sh

# Activate conda environment in the entrypoint
ENTRYPOINT ["conda", "run", "-n", "qwen_env", "/app/start.sh"]

# Expose ports
EXPOSE 8000 8001

# Set healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1
