FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git, wget, and unzip
RUN apt-get update && apt-get install -y git wget unzip && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create model directory structure
RUN mkdir -p /app/backbones/efficientnet_pytorch

# Clone the repository for the model code
RUN git clone https://github.com/megvii-research/CADDM.git caddm

# Create directory for efficientnet weights
RUN mkdir -p /data/efficientnet

# Download the EfficientNet weights
RUN wget -O /data/efficientnet/efficientnet-b4-6ed6700e.pth \
    https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth

# Download model weights (using gdown for Google Drive)
RUN pip install --no-cache-dir gdown
RUN mkdir -p /app/checkpoints
RUN gdown --id 1JNMI4RGssgCOl9t05jkUa6imnw5XR5id -O /app/checkpoints/checkpoints_pretrain.zip && \
    unzip /app/checkpoints/checkpoints_pretrain.zip -d /app/checkpoints && \
    rm /app/checkpoints/checkpoints_pretrain.zip

# Find and move weights files to the expected location
RUN find /app/checkpoints -name "*.pkl" -type f -exec cp {} /app/checkpoints/ \; && \
    ls -la /app/checkpoints/

# Copy necessary files from the repository
RUN cp -r caddm/backbones/* /app/backbones/ && \
    cp -r caddm/detection_layers /app/ && \
    cp -r caddm/lib /app/

# Set environment variables
ENV MODEL_PORT=5006
ENV USE_GPU=true

# Expose port
EXPOSE 5006

# Run the application
CMD ["python", "app.py"]