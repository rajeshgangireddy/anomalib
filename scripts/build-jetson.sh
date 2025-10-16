#!/bin/bash

# Build script for Anomalib Docker container on NVIDIA Jetson Orin
# This script builds the Docker image optimized for Jetson hardware

set -e

echo "🚀 Building Anomalib Docker image for NVIDIA Jetson Orin..."

# Check if we're running on a Jetson device
if ! command -v jetson_release &> /dev/null; then
    echo "⚠️  Warning: jetson_release command not found. Make sure you're running this on a Jetson device."
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Container Runtime is available
if ! docker info | grep -q nvidia; then
    echo "❌ NVIDIA Container Runtime not found. Please install nvidia-docker2."
    exit 1
fi

# Set image name and tag
IMAGE_NAME="anomalib-jetson"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Build the Docker image
echo "🔨 Building Docker image: ${FULL_IMAGE_NAME}"
docker build -f Dockerfile.jetson -t ${FULL_IMAGE_NAME} .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    echo "📦 Image name: ${FULL_IMAGE_NAME}"
    
    # Show image size
    echo "📊 Image details:"
    docker images ${IMAGE_NAME}
    
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Run the container: ./scripts/run-jetson.sh"
    echo "   2. Or use Docker Compose: docker-compose -f docker-compose.jetson.yml up"
    echo "   3. Or run interactively: docker run -it --runtime=nvidia --gpus all ${FULL_IMAGE_NAME}"
else
    echo "❌ Docker build failed!"
    exit 1
fi