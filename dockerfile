# Use an official lightweight base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python, pip, and necessary dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip git wget && \
    apt-get clean

# Install Python dependencies (latest versions)
RUN pip3 install --no-cache-dir torch torchvision torchaudio transformers accelerate bitsandbytes fastapi uvicorn scipy

# Create directories for the app
WORKDIR /app

# Copy the API and LLM scripts to the container
COPY run_llm.py /app/run_llm.py
COPY api.py /app/api.py

# Expose port 8000 for the API
EXPOSE 8000

# Run the FastAPI server when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]