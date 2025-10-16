# Anomalib on NVIDIA Jetson Orin

This directory contains Docker configurations and scripts to run Anomalib on NVIDIA Jetson Orin devices. The setup is optimized for the ARM64 architecture and CUDA capabilities of the Jetson platform.

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA Jetson Orin (Orin NX, Orin Nano, or AGX Orin)
- At least 8GB of RAM (16GB+ recommended for larger models)
- 32GB+ storage space for Docker images and datasets

### Software Requirements
- JetPack 5.0+ (Ubuntu 20.04 based)
- Docker 20.10+
- NVIDIA Container Runtime (nvidia-docker2)
- Git

## 🚀 Quick Start

### 1. Install Prerequisites

First, ensure your Jetson device has JetPack installed and Docker with NVIDIA runtime support:

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA runtime
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### 2. Clone Anomalib Repository

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib
```

### 3. Build the Docker Image

```bash
# Make scripts executable
chmod +x scripts/build-jetson.sh scripts/run-jetson.sh

# Build the Docker image (this may take 30-60 minutes)
./scripts/build-jetson.sh
```

### 4. Run Anomalib Container

```bash
# Interactive mode with bash shell
./scripts/run-jetson.sh

# Start Jupyter Lab server
./scripts/run-jetson.sh --jupyter

# Run in detached mode
./scripts/run-jetson.sh --detach

# Run a specific command
./scripts/run-jetson.sh --command "python -c 'import anomalib; print(anomalib.__version__)'"
```

## 🐳 Docker Usage Options

### Option 1: Using Build Scripts (Recommended)

The provided scripts handle all the complexity:

```bash
# Build image
./scripts/build-jetson.sh

# Run container
./scripts/run-jetson.sh
```

### Option 2: Using Docker Compose

```bash
# Start the main service
docker-compose -f docker-compose.jetson.yml up anomalib-jetson

# Start with Jupyter Lab
docker-compose -f docker-compose.jetson.yml up anomalib-jupyter

# Run in background
docker-compose -f docker-compose.jetson.yml up -d
```

### Option 3: Direct Docker Commands

```bash
# Build image
docker build -f Dockerfile.jetson -t anomalib-jetson:latest .

# Run container
docker run -it --runtime=nvidia --gpus all \
  -v $(pwd)/datasets:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/pre_trained:/workspace/models \
  anomalib-jetson:latest
```

## 💾 Data and Volume Management

The Docker setup automatically mounts the following directories:

- `./datasets` → `/workspace/data` (Your datasets)
- `./results` → `/workspace/results` (Training and inference results)
- `./pre_trained` → `/workspace/models` (Pre-trained model weights)
- `.` → `/workspace/anomalib` (Anomalib source code for development)

## 🧪 Running Anomalib Examples

### 1. Using the CLI

```bash
# Inside the container
cd /workspace/anomalib

# Train a PatchCore model on MVTec AD bottle dataset
anomalib train --model patchcore --data mvtecad --data.category bottle

# Run inference
anomalib predict --model patchcore --data mvtecad --data.category bottle \
  --ckpt_path results/patchcore/mvtecad/bottle/latest/weights/lightning/model.ckpt
```

### 2. Using Python API

```bash
# Start Python inside the container
python3

# Run the following Python code:
```

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
datamodule = MVTecAD(category="bottle")
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)

# Test the model
predictions = engine.test(datamodule=datamodule, model=model)
```

### 3. Using Jupyter Notebooks

```bash
# Start Jupyter Lab
./scripts/run-jetson.sh --jupyter

# Open browser and go to http://localhost:8888
# Navigate to examples/notebooks/ for example notebooks
```

## 🔧 Performance Optimization

### Memory Management

The Jetson Orin has limited memory compared to desktop GPUs. To optimize performance:

```python
# Reduce batch size in your configs
batch_size = 8  # or even smaller for Orin Nano

# Use smaller image sizes
image_size = [256, 256]  # instead of 512x512

# Enable gradient checkpointing for memory efficiency
# This is automatically handled in the Docker image
```

### Model Selection

Some models work better on Jetson than others:

- **Recommended**: PatchCore, STFPM, FastFlow (good balance of accuracy and speed)
- **Fast**: PaDiM, SPADE (lower memory usage)
- **Advanced**: Efficient-AD (specifically designed for edge devices)

### TensorRT Optimization

The Docker image includes TensorRT support for further acceleration:

```python
# Export to ONNX then optimize with TensorRT
# This can provide 2-5x speedup on Jetson
anomalib export --model patchcore --export_mode torch --ckpt_path model.ckpt
# Then use NVIDIA's tools to convert ONNX to TensorRT
```

## 🐛 Troubleshooting

### Common Issues

1. **"nvidia-docker runtime not found"**
   ```bash
   # Reinstall nvidia-docker2 and restart Docker
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Out of memory errors**
   ```bash
   # Reduce batch size or use smaller models
   # Check available memory: nvidia-smi
   ```

3. **Slow build times**
   ```bash
   # The initial build can take 30-60 minutes on Jetson
   # Subsequent builds use Docker cache and are much faster
   ```

4. **Permission issues**
   ```bash
   # Make sure scripts are executable
   chmod +x scripts/*.sh
   
   # Fix Docker permissions
   sudo usermod -aG docker $USER
   # Then logout and login again
   ```

### Performance Monitoring

Monitor your Jetson's performance while running Anomalib:

```bash
# Check GPU utilization
nvidia-smi

# Check system resources
htop

# Check Docker container resources
docker stats anomalib-jetson-orin
```

## 📊 Benchmarks

Expected performance on different Jetson Orin variants:

| Model | Jetson Orin Nano | Jetson Orin NX | AGX Orin |
|-------|------------------|-----------------|----------|
| PaDiM | ~15 FPS (256x256) | ~25 FPS | ~35 FPS |
| PatchCore | ~8 FPS | ~15 FPS | ~25 FPS |
| STFPM | ~12 FPS | ~20 FPS | ~30 FPS |

*Benchmarks are approximate and depend on model configuration and dataset complexity.*

## 🔄 Updates and Maintenance

### Updating Anomalib

```bash
# Pull latest changes
git pull origin main

# Rebuild Docker image
./scripts/build-jetson.sh

# The build process will use cached layers where possible
```

### Cleaning Up

```bash
# Remove containers
docker-compose -f docker-compose.jetson.yml down

# Remove unused images and containers
docker system prune -a

# Remove specific Anomalib image
docker rmi anomalib-jetson:latest
```

## 📝 Development

### Modifying the Docker Setup

The Docker configuration files are:

- `Dockerfile.jetson` - Main Dockerfile optimized for Jetson
- `docker-compose.jetson.yml` - Docker Compose configuration
- `scripts/build-jetson.sh` - Build script with validation
- `scripts/run-jetson.sh` - Run script with options

### Adding Custom Dependencies

Edit `Dockerfile.jetson` to add your custom dependencies:

```dockerfile
# Add your custom packages after line 45
RUN python3 -m pip install your-custom-package
```

### Contributing

If you improve the Jetson setup, please:

1. Test on multiple Jetson variants if possible
2. Update this README with any changes
3. Submit a pull request with clear description

## 📞 Support

For issues specific to Jetson deployment:

1. Check the [troubleshooting section](#-troubleshooting) above
2. Open an issue on the [Anomalib GitHub repository](https://github.com/open-edge-platform/anomalib/issues)
3. Include your Jetson model, JetPack version, and error details

For general Anomalib questions, refer to the main [Anomalib documentation](https://anomalib.readthedocs.io/).

---

**Happy anomaly detection on your Jetson Orin! 🚀**