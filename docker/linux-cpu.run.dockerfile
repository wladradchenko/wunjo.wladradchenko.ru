FROM ubuntu:22.04 AS base

# Copy application
RUN mkdir -p /wunjo/portable
COPY portable/aifc.py /wunjo/portable/aifc.py
COPY portable/sunau.py /wunjo/portable/sunau.py
COPY portable/pyproject_gpu.toml /wunjo/portable/pyproject.toml
COPY portable/src /wunjo/portable/src
COPY requirements_gpu.txt /wunjo/requirements_gpu.txt

# Alternatives source
# RUN sed -i 's|http://archive.ubuntu.com|http://mirror.yandex.ru|g' /etc/apt/sources.list && \
#     sed -i 's|http://security.ubuntu.com|http://mirror.yandex.ru|g' /etc/apt/sources.list

# Updating and installing dependencies, including Python 3.10 if not present
RUN apt-get update && \
    apt-get install -y git libgl1 libglib2.0-0 libsndfile1 ffmpeg python3 python3-pip python3-venv wget && \
    apt-get clean && \
    cd /wunjo && \
    python3 -m pip install --upgrade pip wheel setuptools

# Installing dependencies and creating a virtual environment
RUN cd /wunjo && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install -r requirements_cpu.txt && \
    rm -rf ~/.cache/pip

# Setting environment variables
ENV DEBUG=False

# Set environment variable for WUNJO_HOME
ENV WUNJO_HOME=/

# Create the necessary directories for the license
RUN mkdir -p $WUNJO_HOME/.cache/wunjo/setting

# Opening a port for an application
EXPOSE 8000

# Working directory
WORKDIR /wunjo

# Launching the application
RUN cd /wunjo/portable && . ../venv/bin/activate && briefcase dev
ENTRYPOINT ["briefcase", "dev"]

# RUN
# docker build -t wunjo-cpu -f docker/linux-cpu.run.dockerfile .
# docker run -d -p 8000:8000 --restart unless-stopped --name wunjo-cpu-container wunjo-cpu
