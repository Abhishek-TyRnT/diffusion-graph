#!/bin/bash

IMAGE_NAME="${IMAGE_NAME:-diffusion_graph}"
TAG="${TAG:-dev}"
CONTAINER_NAME="${CONTAINER_NAME:-diffusion_graph_${USER}}"  # Use $USER for the current user
VOLUME_NAME="${VOLUME_NAME:-llvm-project}"

user=$(whoami)
build() {
    docker build -t ${IMAGE_NAME}:${TAG} \
        --build-arg username=${user} \
        --build-arg UID=$(id -u) \
        -f docker/Dockerfile .
}

run() {
    if docker volume ls --format '{{.Name}}' | grep -w "$VOLUME_NAME" > /dev/null; then
        echo "Volume '${VOLUME_NAME}' exists."
    else
        docker volume create ${VOLUME_NAME}
    fi

    if docker ps -a --format '{{.Names}}' | grep -w "$CONTAINER_NAME" > /dev/null; 
    then
        docker start -i ${CONTAINER_NAME}
    else
        docker run -it -v $(pwd):/home/${user}/diffusion-project \
                -v $VOLUME_NAME:/llvm-build \
                --name $CONTAINER_NAME \
                --user $(id -u) \
                -w /home/${user} \
                --gpus all \
                ${IMAGE_NAME}:${TAG} \
                bash
    fi

}

stop() {
    if docker ps -a --format '{{.Names}}' | grep -w "$CONTAINER_NAME" > /dev/null; 
    then
        docker rm $CONTAINER_NAME
    else
        echo "$CONTAINER_NAME doesn't exist"
    fi
}

$1