# Build Guide

This guide explains how to use the `build.sh` script to build Marie AI Docker images.

## Overview

The `build.sh` script provides an automated way to build Marie AI Docker images with support for multiple profiles and configurable versions.

## Prerequisites

- Docker installed and running
- Git (for VCS operations)
- Build context directories: `patches/`, `wheels/`, `Dockerfiles/`

## Available Profiles

| Profile | Description | Output Image |
|---------|-------------|--------------|
| `marie-gateway-cpu` | MarieAI Gateway (CPU optimized) | `marieai/marie-gateway:{VERSION}-cpu` |
| `marie-cuda` | MarieAI Core (CUDA/GPU support) | `marieai/marie:{VERSION}-cuda` |
| `all` | Build all profiles | Both images above |

## Usage

### Interactive Mode

Run the script without arguments for an interactive menu:
```
./build.sh
```
### Command Line Usage
Build specific profile:

```shell
./build.sh marie-gateway-cpu
./build.sh marie-cuda
./build.sh all
./build.sh 1
./build.sh 2
```

Build with custom version:

```shell
./build.sh --version 4.1.0 marie-cuda
./build.sh -v 4.1.0 all
MARIE_VERSION=4.1.0 ./build.sh marie-gateway-cpu
MARIE_VERSION=4.2.0 ./build.sh marie-cuda
```
Get help:
```shell
./build.sh --help
./build.sh -h
```


## Version Management
Default version is 4.0.0
Set custom versions using:
- Command line: ./build.sh --version 4.2.0 marie-cuda
- Environment variable: export MARIE_VERSION=4.2.0
- Inline: MARIE_VERSION=4.2.0 ./build.sh marie-cuda

## Examples
Build single profile:

```shell
./build.sh marie-cuda
```
Build all profiles:

```shell
./build.sh all
```
Build with custom version:
```shell
./build.sh --version 4.1.0 marie-cuda
```


## Build Process
The script performs these steps:
1. Prerequisites Check: Validates Docker installation
2. Post-commit Hook: Executes ./hooks/post-commit if available
3. Docker Build: Builds with no cache, host networking, automatic build arguments
4. Verification: Confirms successful image creation

## Build Arguments
The script automatically sets these Docker build arguments:
- PIP_TAG=standard
- VCS_REF=git_commit_hash
- BUILD_DATE=iso8601_timestamp
- MARIE_VERSION=version
- TARGETPLATFORM=linux/amd64

## Output
After successful builds you will have images like:
- marieai/marie-gateway:4.0.0-cpu
- marieai/marie:4.0.0-cuda

## Troubleshooting
Common issues:
1. Docker not found: Ensure Docker is installed and in PATH
2. Dockerfile not found: Check that the Dockerfile exists at the specified path
3. Build context missing: Ensure patches/ and wheels/ directories exist
4. Permission denied: Make sure the script has execute permissions with chmod +x build.sh

