## G5 Annotator Services with Docker Compose

### Overview

This documentation describes how to use the `docker-compose.g5-annotators.yml` file to deploy and manage the `G5 Annotator` and `G5 LLM Annotator` services. These services are essential components of the Marie-AI platform for advanced document analysis. This setup is designed for a production-like environment that leverages GPU acceleration.

### Prerequisites

Before you begin, ensure your system meets the following requirements:

1.  **System Requirements**:
    *   **Operating System**: Linux (x86\_64)
    *   **GPU**: An NVIDIA GPU with CUDA support is required as the services are configured to use a CUDA-enabled image.
    *   **Storage**: Sufficient disk space for Docker images and Marie-AI models (at least 100GB recommended).

2.  **Software Dependencies**:
    *   **Docker Engine**
    *   **Docker Compose**
    *   **NVIDIA Container Toolkit**: Required for GPU support in Docker.

3.  **Directory and Model Setup**:
    *   The Marie-AI repository should be cloned to your local machine.
    *   The required directory structure and symbolic links must be in place as described in the main `bootstrap.md` guide. Specifically, ensure the `/mnt/data/marie-ai` directory exists and contains the `config` and `model_zoo` directories.

### File Location

Place the `docker-compose.g5-annotators.yml` file inside the `Dockerfiles/` directory of your `marie-ai` project. This keeps it organized with the other Docker-related configuration files.

### How to Use

Follow these steps to manage the G5 annotator services:

#### 1. Starting the Services

To start both the `annotator-g5` and `annotator-g5-llm` services, open a terminal, navigate to the root of the `marie-ai` project, and run the following command:

```shell script
docker compose -f ./Dockerfiles/docker-compose.g5-annotators.yml up -d
```


*   The `-f` flag specifies the path to your compose file.
*   The `-d` flag runs the containers in detached mode, meaning they will run in the background.

#### 2. Monitoring the Services

You can view the logs for each service to monitor its status and check for any errors.

To view the logs for the G5 annotator:

```shell script
docker logs -f marie-annotator-g5-server
```


To view the logs for the G5 LLM annotator:

```shell script
docker logs -f marie-annotator-g5-llm-server
```


*   The `-f` flag follows the log output in real-time. Press `Ctrl+C` to exit.

To check the status of all running containers:

```shell script
docker ps
```


#### 3. Stopping the Services

To stop the G5 annotator services, run the following command from the project root:

```shell script
docker compose -f ./Dockerfiles/docker-compose.g5-annotators.yml down
```


This command will gracefully stop and remove the containers defined in the file.

***