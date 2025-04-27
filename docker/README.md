# Wunjo - Docker Installation Guide

This guide will help you set up and run Wunjo using Docker with NVIDIA GPU support. 

**Note** that the Dockerfile already contains the necessary run commands at the bottom of the file, 
so you don't need to remember them - just reference them when needed (see RUN section).

## Prerequisites

- Requires an NVIDIA GPU with up-to-date drivers and CUDA 12.x installed. (Only NVIDIA hardware is supported.)
- Docker installed
- Git installed

### Install Docker

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker  # Refresh group permissions
```

#### Windows:
1. Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Enable WSL 2 backend (recommended)
3. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/wladradchenko/wunjo.wladradchenko.ru.git
cd wunjo.wladradchenko.ru
```

2. Build the Docker image (this may take some time):
```bash
docker build -t wunjo-run -f docker/linux-cuda-12.4.run.dockerfile .
```

For faster build with buildkit (recommended):
```bash
DOCKER_BUILDKIT=1 docker build --gpus all -t wunjo-run -f docker/linux-cuda-12.4.run.dockerfile .
```

3. Run the container:
```bash
docker run --gpus all -p 8000:8000 wunjo-run
```

For detached mode (running in background):
```bash
docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name wunjo-container wunjo-run
```

## Accessing the Application

After successful launch, open your web browser and navigate to:
```
http://localhost:8000
```

## Troubleshooting

1. **GPU not detected**: Ensure you have NVIDIA drivers installed and Docker has GPU access
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

2. **Port conflict**: If port 8000 is in use, change to another port (e.g., 8001):
   ```bash
   docker run --gpus all -p 8001:8000 wunjo-run
   ```

3. **CUDA version mismatch**: Check your NVIDIA driver supports CUDA 12.4

## Updating

To update to the latest version:
```bash
git pull origin main
docker build -t wunjo-run -f docker/linux-cuda-12.4.run.dockerfile .
docker stop wunjo-container
docker rm wunjo-container
docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name wunjo-container wunjo-run
```

For more information, visit the [documentation](https://github.com/wladradchenko/wunjo.wladradchenko.ru/wiki).
