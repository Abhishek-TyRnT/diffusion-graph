FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

ARG username
ARG UID
RUN apt update && \
    apt install -y clang \
                    cmake \
                    build-essential \
                    python3-pip \
                    sudo \
                    git \
                    ccache \
                    libhdf5-dev

COPY requirements.txt /opt/requirements.txt
WORKDIR /opt
RUN pip3 install -r requirements.txt
RUN useradd -u ${UID} ${username} && \
    usermod -aG sudo ${username} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV TORCH_MLIR_PATH="/home/${username}/vllm-project/build/external/torch-mlir/python_packages/torch_mlir"
ENV VLLM_COMPILER="/home/${username}/vllm-project/build/python"
ENV VLLM_SDK="/home/${username}/vllm-project/vllm_graph"
RUN mkdir /llvm-build && \
    chown ${username} /llvm-build && \
    mkdir /home/${username} && \
    chown ${username} /home/${username} && \
    echo "#!/bin/bash \nsource /home/${username}/vllm-project/scripts/build" >> /home/${username}/.bashrc && \
    echo "export PYTHONPATH=${TORCH_MLIR_PATH}:${VLLM_COMPILER}:${VLLM_SDK}" >> /home/${username}/.bashrc && \
    echo "export PATH=$PATH:/home/${username}/vllm-project/build/bin" >> /home/${username}/.bashrc && \
    chown ${username} /home/${username}/.bashrc && \
    chmod +x /home/${username}/.bashrc

