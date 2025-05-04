FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y git wget curl unzip findutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/HongguLiu/Deepfake-Detection.git faceforensics_plus_plus

# Create directory for model weights
RUN mkdir -p /app/models

# Install latest version of gdown
RUN pip install --no-cache-dir --upgrade gdown

# Download individual files with their direct IDs, using recommended format
# Download deepfake_c0_xception.pkl
RUN echo "Downloading deepfake_c0_xception.pkl..." && \
    gdown 1eHRN117X0loEff7EBk1mGMJeGbGKsd7m -O /app/models/deepfake_c0_xception.pkl --no-cookies

# Download ffpp_c23.pth
RUN echo "Downloading ffpp_c23.pth..." && \
    gdown 1Zi66nWADRIgS8B01Doa1aUvZmH57bllP -O /app/models/ffpp_c23.pth --no-cookies

# Download ffpp_c40.pth
RUN echo "Downloading ffpp_c40.pth..." && \
    gdown 1jyvW3GCmYqS6gItcw50ot2GI2LrZACwc -O /app/models/ffpp_c40.pth --no-cookies

# If no models were downloaded, fail the build
RUN if [ ! -f "/app/models/deepfake_c0_xception.pkl" ] && \
    [ ! -f "/app/models/ffpp_c23.pth" ] && \
    [ ! -f "/app/models/ffpp_c40.pth" ]; then \
        echo "ERROR: No model files were downloaded!" && \
        exit 1; \
    fi

# Create symlink to default model
RUN if [ -f "/app/models/deepfake_c0_xception.pkl" ]; then \
        ln -sf /app/models/deepfake_c0_xception.pkl /app/models/model.pth; \
    elif [ -f "/app/models/ffpp_c23.pth" ]; then \
        ln -sf /app/models/ffpp_c23.pth /app/models/model.pth; \
    elif [ -f "/app/models/ffpp_c40.pth" ]; then \
        ln -sf /app/models/ffpp_c40.pth /app/models/model.pth; \
    else \
        echo "ERROR: Failed to create model.pth symlink" && \
        exit 1; \
    fi

# Verify model files exist and have content
RUN ls -la /app/models/ && \
    echo "Model file sizes:" && \
    du -h /app/models/*

# Set environment variables
ENV MODEL_PORT=5007
ENV USE_GPU=true
ENV MODEL_PATH=/app/models/model.pth

# Expose port
EXPOSE 5007

# Run the application
CMD ["python", "app.py"]