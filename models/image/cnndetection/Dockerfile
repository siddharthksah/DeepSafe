FROM python:3.9-slim

WORKDIR /app
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only PyTorch for arm64
RUN pip install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt


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
RUN git clone https://github.com/PeterWang512/CNNDetection.git cnndetection

# Download model weights with the correct URL
RUN mkdir -p cnndetection/weights && \
    wget -O cnndetection/weights/blur_jpg_prob0.5.pth \
    https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1

# Set environment variables
ENV MODEL_PORT=5008
ENV USE_GPU=true
ENV MODEL_PATH="cnndetection/weights/blur_jpg_prob0.5.pth"

# Expose port
EXPOSE 5008

# Run the application
CMD ["python", "app.py"]