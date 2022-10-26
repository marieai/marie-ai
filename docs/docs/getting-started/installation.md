---
sidebar_position: 1
---

# Installation

## Prerequisites
* Linux
* Python 3.8
* Pytorch 1.11.0+cu113  
* CUDA 11.3.1


## Environment Setup

:::note

If you are experienced with PyTorch and have already installed it, just skip this part and jump to the next section. Otherwise, you can follow [these steps](#installation-steps) for the preparation.

:::


##  Using a Python virtual environment

```bash
mkdir ~/marie-ai
cd ~/marie-ai
```


From inside this directory, create a virtual environment using the Python venv module:

```shell
sudo apt install python3.10-venv

python -m venv .env
```

Alernativelly you can have shared virtual environment

```bash
python3 -m venv ~/environment/marie
```
This will require you to create a link a sympolic link `.env ` taht point to the real environment `~/environment/marie`


You can jump in and out of your virtual environment with the activate and deactivate scripts:

```shell
# Activate the virtual environment
source .env/bin/activate

# Deactivate the virtual environment
source .env/bin/deactivate
```

## Using Conda

TODO : conda have not been tested but in should work.

```shell
conda create -n pytorch python=3.10
```

```shell
conda activate pytorch
```

### Verify Pytorch Install

```python
python -c "import torch; print(torch.__version__)"
```

##  Installation Steps

There are number of different ways that this project can be setup.

### From source

If you wish to run and develop `Marie-AI` directly, install it from source:

```shell
git clone https://github.com/gregbugaj/marie-ai.git
cd marie-ai
git checkout develop

# "-v" increases pip's verbosity.
# "-e" means installing the project in editable mode,
# That is, any local modifications on the code will take effect immediately

pip install -r requirements.txt
pip install -v -e .

```

### Additional dependencies

```shell
sudo apt-get install libpq-dev python-dev-is-python3
```

```shell
python3 -m pip install -U 'git+https://github.com/facebookresearch/fvcore'
```


```shell
git clone https://github.com/pytorch/fairseq.git
cd fairseq 
python setup.py build install
```

Detectron2 install 

```shell
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

```


Common Installation Issues

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
fairseq 0.12.2 requires hydra-core<1.1,>=1.0.7, but you have hydra-core 1.2.0 which is incompatible.
fairseq 0.12.2 requires omegaconf<2.1, but you have omegaconf 2.2.3 which is incompatible.

```shell
 pip uninstall hydra-core
 pip uninstall omegacon
 pip uninstall fairseq
```

Install `Detectron2` and then `fairseq`

TODO: This needs better setup
```shell
sudo chown greg:greg /var/log/marie/
sudo mkdir -p /var/log/marie
```
### Marie-AI as a dependency  
If you use Marie-AI as a dependency or third-party package, install it with pip:

```shell
pip install 'marie-ai>=2.4.0'
```

### Verify the installation

We provide a method to verify the installation via inference demo, depending on your installation method.

```
TODO GRADIO LINK 
```

Also, you can run the following codes in your Python interpreter:

```python
  from marie.executor import NerExtractionExecutor
  from marie.utils.image_utils import hash_file

  # setup executor
  models_dir = ("/mnt/data/models/")
  executor = NerExtractionExecutor(models_dir)

  img_path = "/tmp/sample.png"
  checksum = hash_file(img_path)

  # invoke executor
  docs = None
  kwa = {"checksum": checksum, "img_path": img_path}
  results = executor.extract(docs, **kwa)

  print(results)

```


### Install on CPU-only platforms

Marie-AI can be built for CPU-only environment. In CPU mode you can train, test or inference a model.
However, there might be limitations of what operations can be used.

```shell
DOCKER_BUILDKIT=1 docker build . -f Dockerfile-cpu -t gregbugaj/marie-icr:2.5 --no-cache  
```

## Docker with GPU Support

### Inference on the gpu
Install following dependencies to ensure docker is setup for GPU processing.

* [Installing Docker and The Docker Utility Engine for NVIDIA GPUs](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

After the installation we can validate the setup with :

[CUDA and cuDNN images from gitlab.com/nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags?page=2&ordering=last_updated&name=11.3)


```shell
#### Test nvidia-smi with the official CUDA image
docker run --gpus all nvidia/cuda:11.3.1-runtime-ubuntu20.04 nvidia-smi
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvidia/cuda:11.3.1-runtime-ubuntu20.04 nvidia-smi
```


### Building container
If we have properly configured our environment you should be able to build the container locally.

```shell
DOCKER_BUILDKIT=1 docker build . -f Dockerfile -t gregbugaj/marie-icr:2.5-cuda --no-cache 
```

## Container maintenance

Marie comes with number of utilities for managing the containers. 

:::note

When deployed with [Kubernetes Control Plane](/docs/getting-started/deployment/control-plane#kubernetes), 
some of  will be redundant.

:::


```text
(marie) greg@xpredator:~/dev/marie-ai$ tree docker-util/
docker-util/
├── container.sh
├── destroy.sh
├── exec.sh
├── id
├── id.github
├── monitor.sh
├── README.md
├── restart.sh
├── run-all.sh
├── run-interactive-cpu.sh
├── run-interactive-gpu.sh
├── run.sh
├── service.env
├── stop.sh
├── tail-log.sh
├── update.sh
└── version.sh

```

|Feature|Interactive|Description|
|---|--|-----------------------------------------------------|
|container.sh|No| Display container information from `id` file          |
|destroy.sh|Yes|Destroy currently installed container and remove image  |
|exec.sh|No| Login into current container          |
|id|No| **ID** file describing  the container          |
|monitor.sh|No| Docker container monitoring via cAdvisor, DCGM-Exporter, Grafana Loki and Promtail  (convert to docker-compose)  |
|restart.sh|No| Restart all containers         |
|run-all.sh|No| Run all containers         |
|run-interactive-cpu.sh|Yes| Run CPU container in interactive mode         |
|run-interactive-gpu.sh|Yes| Run GPU container in interactive mode. Start container with GPU-ID 0 `./run-interactive-gpu.sh 0`        |
|stop.sh|No| Stop all containers         |
|tail-log.sh|No| Tail logs from console         |
|update.sh|No| Update container to version specified in `id` file         |
|version.sh|No| Display `marie-ai`  version       |

:::note Security/Audit

* This script must not be run as root, run under 'docker user' account.

* Scripts will redirect the output of the current script to the system logger. 
```shell
exec 1> >(exec logger -s -t "${CONTAINER_NAME} [${0##*/}]") 2>&1
echo " container : ${CONTAINER_NAME}"
```

:::

### Useful docker commands

Stop running `marie-ai` containers and remove volumes

```shell
docker container stop $(docker container ls -q --filter name='marie*') && docker system prune -af --volumes
```

Tail logs
```shell
docker logs  marie-icr-0  -f --since 0m
```




## Common issues

### Segmentation fault

There is a segmentation fault happening with `opencv-python==4.5.4.62` switching to `opencv-python==4.5.4.60` fixes the issue. 
[connectedComponentsWithStats produces a segfault ](https://github.com/opencv/opencv-python/issues/604)

```
pip install opencv-python==4.5.4.60
```


### Missing convert_namespace_to_omegaconf

Install `fairseq` from source, the release version is  missing `convert_namespace_to_omegaconf`

```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install -r requirements.txt
python setup.py build develop
```


### `distutils` has no attribute version

If you receive following error :

```
AttributeError: module 'distutils' has no attribute 'version'
```

Using following version of `setuptools` will work.

```
python3 -m pip install setuptools==59.5.0
```


### CUDA capability sm_86 is not compatible with the current PyTorch installation

Building GPU version of the framework requires `1.10.2+cu113`. 

If you encounter following error that indicates that we have a wrong version of PyTorch / Cuda

```
1.11.0+cu102
Using device: cuda

/opt/venv/lib/python3.8/site-packages/torch/cuda/__init__.py:145: UserWarning: 
NVIDIA GeForce RTX 3060 Laptop GPU with CUDA capability sm_86 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA GeForce RTX 3060 Laptop GPU GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))

```



### References
[Docker overview](https://docs.docker.com/get-started/overview/)
