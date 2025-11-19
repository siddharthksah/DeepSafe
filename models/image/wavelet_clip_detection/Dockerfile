# Use a specific Python version
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y git wget build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy with compatible version for scipy
RUN pip install --no-cache-dir numpy==1.22.0

# Install PyTorch
RUN pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone the Wavelet-CLIP repository
RUN git clone https://github.com/lalithbharadwajbaru/Wavelet-CLIP.git model_code

# Create directory for model weights
RUN mkdir -p model_code/weights

# Download model weights from Google Drive and handle subfolder
RUN gdown --folder https://drive.google.com/drive/folders/1Z7pH9KPQbx1TrMqap2y9Op6OUO9SHP_D -O model_code/weights/ && \
    ls -la model_code/weights/ && \
    # Move the files from the subfolder to the main weights directory
    mv model_code/weights/wavelet_clip_detection/* model_code/weights/ && \
    # Verify the files were moved
    ls -la model_code/weights/

# Create __init__.py files to make directories proper Python packages
RUN touch model_code/__init__.py && \
    touch model_code/training/__init__.py && \
    touch model_code/training/detectors/__init__.py && \
    touch model_code/training/metrics/__init__.py && \
    touch model_code/training/loss/__init__.py

# Copy the application file
COPY app.py .

# Environment Variables
ENV MODEL_PORT=5003
ENV MODEL_PATH=model_code/weights/clip_wavelet_best.pth
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="600"
ENV PYTHONPATH=/app/model_code:/app/model_code/training:$PYTHONPATH

# Expose port
EXPOSE ${MODEL_PORT}

# Command to run the application
CMD ["python", "app.py"]