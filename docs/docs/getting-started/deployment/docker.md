---
sidebar_position: 1
---

# Docker
Deployment via docker container


## User and permission setup
The container is setup with `app-svc` account so for that we will setup same account in the host system.


Setting up user, for more info visit [Manage Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user)

```
sudo useradd --comment 'app-svc' --create-home app-svc --shell /bin/bash
sudo usermod -aG docker app-svc
```

Directory structure, this is more of a convention than requirement.

```sh
sudo mkdir -p /opt/containers/apps/marie-ai
sudo mkdir -p /opt/containers/config/marie-ai
```

Change permissions to `app` and `config`

```
cd /opt/containers
sudo chown app-svc:app-svc apps/ -R
sudo chown app-svc:app-svc config/ -R
```

Now that we have our directory and permissions setup we can move on and setup the container.
The easiest way to manage container is by checking out the [Marie-AI project](https://github.com/gregbugaj/marie-ai.git)

```sh
sudo su app-svc

git clone https://github.com/gregbugaj/marie-ai.git
```

## Basic Configuration

:::warning Container Configuration

This is a single node setup up without control-plane.

Complete configuration setup can be found under [Configuration](/docs/category/configuration).

:::

Before we are able to start the container we need to configure few components. 


### Directory layout
This is the intial entrypoint for `models` and `config`, this can be changed by modifying `run.sh` Directories will be mapped as volumes when the container is created.

***TODO : this needs to be configurable via ENV variables***

```
/mnt/data/marie-a
/opt/logs/marie-icr
```


#### Single-GPU

On a single GPU the directory structure should loook like following

```
/mnt/data/marie-ai/
├── config
│   ├── consul
│   ├── executors
│   ├── ocr
│   └── traefik
│       └── log
└── model_zoo
```

#### Multi-GPU 

If the system support muli-gpus then we can configure the system to take advanate of that. Configuration is almost identical to single gpu. 
For each GPU we adappend GPU-ID to the config directory names

```
/mnt/data/marie-ai/
├── config-0
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-1
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-2
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-3
│   ├── consul
│   ├── ocr
│   └── traefik
├── config-template
│   ├── consul
│   ├── ocr
│   └── traefik
└── model_zoo
```

### Required configuration

The easiest way to initialize the configs is by copying them from the project `config` directory.

```
/mnt/data/marie-ai/config/marie.yml
```


## Container management

To manage a container we have a set of scripts located in `marie-ai/docker-util`

:::info Changing container version

To change container version edit `id` file, it should contain a specific version that we like to use [Docker Registry](https://docs.docker.com/registry/).

```
gregbugaj/marie-icr:2.4-cuda
```

This could also include your specific registy as well.

```
localhost:5000/gregbugaj/marie-icr:2.4-cuda
```

:::


### Update container to specific version

```sh
cd marie-ai/docker-util
./update.sh
```


### Starting container


Start single `marie-ai` container in interactive mode with **ALL GPUS** assigned to single instance

```
./run-interactive-gpu.sh
```

You should see initial output :

```
Starting interactive/dev container : marie-icr
GPU_COUNT : 4
GPUS Selected  > all
CONTAINER_NAME > marie-icr
CONTAINER_ID   > 
CONFIG         > config
PORT           > 6000
CONFIG_DIR     > /mnt/data/marie-ai/config
```

