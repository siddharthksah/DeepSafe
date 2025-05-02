#!/bin/bash

# Create required directories
mkdir -p model/cnn_detection/weights
mkdir -p model/hifi_ifdl/weights
mkdir -p frontend/public
mkdir -p frontend/src

# Clone CNNDetection repository
echo "Cloning CNNDetection repository..."
git clone https://github.com/PeterWang512/CNNDetection.git temp_cnn_detection

# Copy necessary files to our directory structure
echo "Copying CNNDetection files to model/cnn_detection..."
cp -r temp_cnn_detection/* model/cnn_detection/

# Download model weights for CNNDetection
echo "Downloading CNNDetection model weights..."
wget -O model/cnn_detection/weights/blur_jpg_prob0.5.pth https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1

# Clone HiFi_IFDL repository
echo "Cloning HiFi_IFDL repository..."
git clone https://github.com/CHELSEA234/HiFi_IFDL.git temp_hifi_ifdl

# Copy necessary files to our directory structure
echo "Copying HiFi_IFDL files to model/hifi_ifdl..."
cp -r temp_hifi_ifdl/* model/hifi_ifdl/

# Download model weights for HiFi_IFDL
echo "Downloading HiFi_IFDL model weights..."
mkdir -p model/hifi_ifdl/weights
mkdir -p model/hifi_ifdl/center

# Note: These URLs might need to be updated with the actual direct download links
wget -O model/hifi_ifdl/weights/HRNet.pth "https://drive.google.com/uc?export=download&id=1YOYDYw-gAR_F8TFM1KSE6iY7YOvdkvsJ"
wget -O model/hifi_ifdl/weights/NLCDetection.pth "https://drive.google.com/uc?export=download&id=1WGR9EbXIK3FEZv3JC26pDRh3DcRq_Vfm"
wget -O model/hifi_ifdl/center/radius_center.pth "https://drive.google.com/uc?export=download&id=1sSCFxkbFNFnKPuYA4tYnDGOCQkjIGXEG"

# Clean up
echo "Cleaning up..."
rm -rf temp_cnn_detection
rm -rf temp_hifi_ifdl

echo "Setup complete!"
echo "You can now run 'docker-compose up -d' to start the services."