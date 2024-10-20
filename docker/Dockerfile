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
                    ccache

COPY requirements.txt /opt/requirements.txt
WORKDIR /opt
RUN pip3 install -r requirements.txt
RUN useradd -u ${UID} ${username} && \
    usermod -aG sudo ${username} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir /llvm-build && \
    chown ${username} /llvm-build && \
    mkdir /home/${username} && \
    chown ${username} /home/${username} && \
    echo "#!/bin/bash \nsource /home/${username}/vllm-project/scripts/build" >> /home/${username}/.bashrc && \
    echo "export PYTHONPATH=/home/${username}/vllm-project/compiler/build/external/torch-mlir/build/python_packages/torch_mlir" >> /home/${username}/.bashrc && \
    echo "export PATH=$PATH:/home/${username}/vllm-project/compiler/build/bin" >> /home/${username}/.bashrc && \
    chown ${username} /home/${username}/.bashrc && \
    chmod +x /home/${username}/.bashrc

