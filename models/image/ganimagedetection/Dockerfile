FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

# Install git and wget
RUN apt-get update && apt-get install -y git wget && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/grip-unina/GANimageDetection.git ganimagedetection

# Download model weights
RUN mkdir -p ganimagedetection/weights && \
    wget -O ganimagedetection/weights/gandetection_resnet50nodown_stylegan2.pth \
    https://www.grip.unina.it/download/prog/GANdetection/weights/gandetection_resnet50nodown_stylegan2.pth

# Set environment variables
ENV MODEL_PORT=5001
ENV USE_GPU=true

# Expose port
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]