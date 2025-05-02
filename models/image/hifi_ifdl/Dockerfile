# Use the PyTorch base image with CUDA support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set timezone to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore

WORKDIR /app

# Install git, wget, and gdown (for Google Drive downloads)
# Also install necessary dependencies including python3-opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget python3-pip python3-dev build-essential python3-opencv \
    && pip install --no-cache-dir gdown \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir yacs matplotlib einops pillow scikit-image tensorboard

# Clone the HiFi_IFDL repository
RUN git clone https://github.com/CHELSEA234/HiFi_IFDL.git hifi_ifdl_code

# Create weights directory
RUN mkdir -p /app/weights

# Download HRNet weights
RUN gdown --id 1BSg39gzlUM_odiiuV5NIJoc8f90JCCqf -O /app/weights/750001.pth

# Download NLCDetection weights and rename to the expected filename
RUN gdown --id 1hELPa0bIyLrnr0a06vmr6GWNNinWB0nf -O /app/weights/NLCD_weights.pth && \
    mv /app/weights/NLCD_weights.pth /app/weights/NLCDetection.pth

# Verify weights are downloaded correctly
RUN ls -l /app/weights

# Patch the code to fix the np.int deprecation
RUN sed -i 's/np.int(/int(/g' /app/hifi_ifdl_code/models/seg_hrnet.py

# Copy the FastAPI application file
COPY app.py .

# Set environment variables
ENV MODEL_PORT=5003
ENV USE_GPU=true
ENV HRNET_WEIGHT_PATH="/app/weights/750001.pth"
ENV NLCD_WEIGHT_PATH="/app/weights/NLCDetection.pth"
# Set PYTHONUNBUFFERED to see logs in real-time
ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 5003

# Command to run the application using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]