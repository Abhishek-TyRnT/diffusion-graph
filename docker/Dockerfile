FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu22.04

ARG username
ARG UID
#TODO: remove h5
RUN apt update && \
    apt install -y clang \
                    build-essential \
                    python3-pip \
                    sudo \
                    git \
                    ccache \
                    curl \
                    unzip \
                    valgrind \
                    libnuma-dev \
                    python3-venv \
                    patchelf


COPY requirements.txt /opt/requirements.txt
WORKDIR /opt
RUN pip3 install -r requirements.txt
RUN useradd -u ${UID} ${username} && \
    usermod -aG sudo ${username} && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN git clone -b v30.1 https://github.com/protocolbuffers/protobuf.git && \
    cd protobuf && \
    git submodule update --init && \
    cmake . -G "Ninja" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF \
    -Dprotobuf_WITH_ZLIB=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    cmake --build . && \
    cmake --build . --target install


ENV TORCH_MLIR_PATH="/home/${username}/diffusion-project/build/external/torch-mlir/python_packages/torch_mlir"
ENV DIFFUSION_GRAPH_COMPILER="/home/${username}/diffusion-project/build/python"
ENV DIFFUSION_GRAPH_SDK="/home/${username}/diffusion-project/diffusion_graph"
RUN mkdir /llvm-build && \
    chown ${username} /llvm-build && \
    mkdir /home/${username} && \
    chown ${username} /home/${username} && \
    echo "#!/bin/bash \nsource /home/${username}/diffusion-project/scripts/build" >> /home/${username}/.bashrc && \
    echo "export PYTHONPATH=${TORCH_MLIR_PATH}:${DIFFUSION_GRAPH_COMPILER}:${DIFFUSION_GRAPH_SDK}" >> /home/${username}/.bashrc && \
    echo "export PATH=$PATH:/home/${username}/diffusion-project//build/bin:/llvm_build/llvm-project/build/bin" >> /home/${username}/.bashrc && \
    chown ${username} /home/${username}/.bashrc && \
    chmod +x /home/${username}/.bashrc

