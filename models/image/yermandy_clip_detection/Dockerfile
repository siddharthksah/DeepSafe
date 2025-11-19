# Use a PyTorch base image
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

WORKDIR /app

# Install git and wget for cloning repo and downloading weights
RUN apt-get update && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone the model repository
# We'll name the cloned directory "model_code" to avoid conflicts
RUN git clone https://github.com/yermandy/deepfake-detection.git model_code

# Download model weights
RUN mkdir -p model_code/weights && \
    wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O model_code/weights/model.ckpt

# Set Hugging Face home to cache pre-trained models inside the container
ENV HF_HOME=/app/huggingface_cache
ENV TRANSFORMERS_CACHE=/app/huggingface_cache
ENV HF_HUB_CACHE=/app/huggingface_cache
RUN mkdir -p $HF_HOME

# Pre-download Hugging Face CLIPProcessor models to be self-sustaining
# This ensures these are part of the Docker image layer
RUN python -c "from transformers import CLIPProcessor; \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14-336')"

# Copy the application file
COPY app.py .

# --- Environment Variables ---
# MODEL_PORT will be set by docker-compose, defaulting here for clarity
ENV MODEL_PORT=5002
ENV MODEL_PATH=model_code/weights/model.ckpt
ENV PRELOAD_MODEL="false"
# MODEL_TIMEOUT: Seconds to keep model loaded if not preloaded
ENV MODEL_TIMEOUT="600"

# Add the cloned repository's root and src to PYTHONPATH
# This allows app.py to import modules from the cloned model_code directory
ENV PYTHONPATH=/app/model_code:/app/model_code/src:$PYTHONPATH

# Expose the port the app runs on
EXPOSE ${MODEL_PORT}

# Command to run the application
CMD ["python", "app.py"]