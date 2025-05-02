#!/bin/bash

# Create required directories
mkdir -p model/cnn_detection/weights
mkdir -p frontend/public
mkdir -p frontend/src

# Clone CNNDetection repository
echo "Cloning CNNDetection repository..."
git clone https://github.com/PeterWang512/CNNDetection.git temp_cnn_detection

# Copy necessary files to our directory structure
echo "Copying CNNDetection files to model/cnn_detection..."
cp -r temp_cnn_detection/* model/cnn_detection/

# Download model weights
echo "Downloading model weights..."
wget -O model/cnn_detection/weights/blur_jpg_prob0.5.pth https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1

# Clean up
echo "Cleaning up..."
rm -rf temp_cnn_detection

echo "Setup complete!"
echo "You can now run 'docker-compose up -d' to start the services."