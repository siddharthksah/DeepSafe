FROM python:3.9-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install uv

WORKDIR /app

# Clone the original repository for model code
RUN git clone https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts.git temp_repo

# Install Python dependencies
RUN uv pip install --system --no-cache \
    torch==1.13.1 \
    torchaudio==0.13.1 \
    flask==2.3.0 \
    soundfile==0.12.1 \
    librosa==0.10.0 \
    numpy==1.24.0

# Create models directory for weights
RUN mkdir -p /app/models

# Copy API wrapper
COPY api.py .

# Download pretrained model (users should mount or download separately)
# Due to size, this should be done via volume mount or manual download
# URL: https://drive.google.com/file/d/15qOi26czvZddIbKP_SOR8SLQFZK8cf8E/view

# Expose port
EXPOSE 8001

ENV MODEL_PORT=8001
ENV PYTHONUNBUFFERED=1

# Run the API
CMD ["python", "api.py"]
