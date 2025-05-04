FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git, wget, unzip, and pip
RUN apt-get update && \
    apt-get install -y git wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/grip-unina/DMimageDetection.git dmimagedetection

# Create weights directory structure
RUN mkdir -p dmimagedetection/weights/Grag2021_progan && \
    mkdir -p dmimagedetection/weights/Grag2021_latent

# Install gdown for Google Drive downloads
RUN pip install --no-cache-dir gdown

# Download model weights from Google Drive
RUN gdown 1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf -O weights.zip && \
    unzip -o weights.zip && \
    # Check the directory structure
    ls -la && \
    # Handle either directory structure
    (if [ -d "weights/weights" ]; then \
        echo "Found nested weights directory" && \
        mv weights/weights/Grag2021_progan/model_epoch_best.pth dmimagedetection/weights/Grag2021_progan/ && \
        mv weights/weights/Grag2021_latent/model_epoch_best.pth dmimagedetection/weights/Grag2021_latent/; \
    else \
        echo "Found standard weights directory" && \
        mv weights/Grag2021_progan/model_epoch_best.pth dmimagedetection/weights/Grag2021_progan/ && \
        mv weights/Grag2021_latent/model_epoch_best.pth dmimagedetection/weights/Grag2021_latent/; \
    fi) && \
    # Clean up
    rm -rf weights weights.zip

# Set environment variables
ENV MODEL_PORT=5005
ENV USE_GPU=true

# Expose port
EXPOSE 5005

# Run the application
CMD ["python", "app.py"]