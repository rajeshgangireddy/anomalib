#!/bin/bash

# Run script for Anomalib Docker container on NVIDIA Jetson Orin
# This script runs the Docker container with proper GPU access and volume mounts

set -e

echo "🚀 Starting Anomalib Docker container on NVIDIA Jetson Orin..."

# Configuration
IMAGE_NAME="anomalib-jetson:latest"
CONTAINER_NAME="anomalib-jetson-orin"
DATA_DIR="$(pwd)/datasets"
RESULTS_DIR="$(pwd)/results"
MODELS_DIR="$(pwd)/pre_trained"

# Create directories if they don't exist
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${MODELS_DIR}"

# Check if image exists
if ! docker image inspect ${IMAGE_NAME} &> /dev/null; then
    echo "❌ Docker image ${IMAGE_NAME} not found!"
    echo "🔨 Please build the image first: ./scripts/build-jetson.sh"
    exit 1
fi

# Stop and remove existing container if running
if docker ps -a --format 'table {{.Names}}' | grep -q ${CONTAINER_NAME}; then
    echo "🛑 Stopping existing container..."
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# Parse command line arguments
COMMAND="bash"
RUN_MODE="interactive"
JUPYTER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --jupyter)
            JUPYTER=true
            COMMAND="jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"
            shift
            ;;
        --command)
            COMMAND="$2"
            RUN_MODE="command"
            shift 2
            ;;
        --detach|-d)
            RUN_MODE="detached"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --jupyter         Start Jupyter Lab server"
            echo "  --command CMD     Run specific command"
            echo "  --detach, -d      Run in detached mode"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Interactive bash shell"
            echo "  $0 --jupyter              # Start Jupyter Lab"
            echo "  $0 --command 'python -c \"import anomalib; print(anomalib.__version__)\"'"
            echo "  $0 --detach               # Run in background"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set up Docker run arguments
DOCKER_ARGS=(
    "--runtime=nvidia"
    "--gpus=all"
    "--name=${CONTAINER_NAME}"
    "-e NVIDIA_VISIBLE_DEVICES=all"
    "-e NVIDIA_DRIVER_CAPABILITIES=all"
    "-e CUDA_VISIBLE_DEVICES=0"
    "-v ${DATA_DIR}:/workspace/data:rw"
    "-v ${RESULTS_DIR}:/workspace/results:rw"
    "-v ${MODELS_DIR}:/workspace/models:rw"
    "-v $(pwd):/workspace/anomalib:rw"
    "-v /etc/localtime:/etc/localtime:ro"
    "-w /workspace"
)

# Add interactive or detached mode
if [[ ${RUN_MODE} == "interactive" ]]; then
    DOCKER_ARGS+=("-it")
elif [[ ${RUN_MODE} == "detached" ]]; then
    DOCKER_ARGS+=("-d")
fi

# Add port mappings if needed
if [[ ${JUPYTER} == true ]]; then
    DOCKER_ARGS+=("-p 8888:8888")
    echo "🪐 Jupyter Lab will be available at: http://localhost:8888"
fi

# Add X11 forwarding for GUI applications (optional)
if [[ -n "${DISPLAY}" ]]; then
    DOCKER_ARGS+=(
        "-e DISPLAY=${DISPLAY}"
        "-e QT_X11_NO_MITSHM=1"
        "-v /tmp/.X11-unix:/tmp/.X11-unix:rw"
    )
fi

echo "🐳 Running Docker container with the following configuration:"
echo "   Image: ${IMAGE_NAME}"
echo "   Container: ${CONTAINER_NAME}"
echo "   Data directory: ${DATA_DIR}"
echo "   Results directory: ${RESULTS_DIR}"
echo "   Models directory: ${MODELS_DIR}"
echo "   Command: ${COMMAND}"
echo ""

# Run the container
docker run "${DOCKER_ARGS[@]}" ${IMAGE_NAME} ${COMMAND}

if [[ ${RUN_MODE} == "detached" ]]; then
    echo ""
    echo "✅ Container started in detached mode"
    echo "🔍 View logs: docker logs -f ${CONTAINER_NAME}"
    echo "🖥️  Execute commands: docker exec -it ${CONTAINER_NAME} bash"
    echo "🛑 Stop container: docker stop ${CONTAINER_NAME}"
fi