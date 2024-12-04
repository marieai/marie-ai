# Uses multi-staged approach to reduce size

# Stage 1
# Use base conda image to reduce time
FROM continuumio/miniconda3:latest AS base-image

# Specify py version
ENV PYTHON_VERSION=3.12
# Install audio-related libraries
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git lfs install

RUN conda create --name marie python=${PYTHON_VERSION} pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
ENV PATH /opt/conda/envs/marie/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Activate the conda env


# Stage 2
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build-image
# FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

COPY --from=base-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH

# Install apt libs
RUN apt-get update && \
    apt-get install -y curl git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*