---
sidebar_position: 1
---

# Contribute to MarieAI(🦊)

# Coding standards

To ensure the readability of our code, we stick to a few conventions:

* We format python files using black.
* For linting, we use flake8.
* For sorting imports, we use isort.


The `setup.cfg` and `pyproject.toml` already contain the proper configuration for these tools. 
If you are working in a modern IDE (such a VSCode), which integrates these tools, the options will be picked up.
If you are working from the command line, you should use these tools form the root of this repository or via a `make` file.



We also provide a .pre-commit-config.yaml file, which enables you to use pre-commit to automatically perform the required linting/formatting actions before you commit your changes. 
You can install it using
```shell
pip install pre-commit
pre-commit install
```

## Downloading large artifacts

Often you will find that for the executor to work, a large file needs to be downloaded first - usually this would be a file with pre-trained model weights. If this is done at the start of the executor, it will lead to really long startup times, or even timeouts, which will frustrate users.
If this file is baked in the docker image (not even possible in all cases), this will create overly large docker images, and prevent users for optimizing storage in cases where they want to run multiple instances of the executor on the same machine, among other things.

The solution in this case is to instruct users how to download the file before starting the executor, and then use that file in their executor. So for this case, you should:

* Add simple copy-pastable instructions on how to download the large files to the readme
* Add instructions on how to specify the path to the file at executor initialization (if needed) and how to mount the file to a Docker container in the readme
* If the file path is not provided, or the file doesn't exist, add an error telling the user that file needs to be downloaded, and pointing them to the readme for further instructions.


## Development environment
If you wish to run and develop `Marie-AI` directly, [install it from source](../installation.md#installing-from-source):

```shell
git clone https://github.com/marieai/marie-ai.git
cd marie-ai
git checkout develop

# "-v" increases pip's verbosity.
# "-e" means installing the project in editable mode,
# That is, any local modifications on the code will take effect immediately

pip install  Cython
pip install pybind11

pip install -r requirements.txt
pip install -v -e .
```

Test your installation by running the following command:

```shell
marie --help
```

### Starting the server

To start the server, run the following command:

```shell
marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml
```


For development purposes, you can also start the storage services in a separate container:
```shell
docker compose  --env-file ./config/.env -f ./Dockerfiles/docker-compose.s3.yml -f ./Dockerfiles/docker-compose.storage.yml --project-directory . up  --build --remove-orphans
```
 

### Memory profiling
Via the `memray` command, you can profile the memory usage of the server :

**Script based profiling:**

```shell
MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai memray run --live ./marie/__main__.py server --start --uses /mnt/data/marie-ai/config/service/marie.yml
```

**Module based profiling:**
```shell
 JINA_MP_START_METHOD=fork PYTHONMALLOC=malloc MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai memray run --follow-fork  -o ~/tmp/memray/marie.bin -m marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml
```

**Profiling with a live remote:**
This is useful as it allows us to observe logs and other information while the server is running.

Server:
```shell
MARIE_DEFAULT_MOUNT=/mnt/data/marie-ai memray run --live-remote -m marie server --start --uses /mnt/data/marie-ai/config/service/marie.yml
```
Remote :
```shell
memray3.10 live
``` 

### Testing for memory leaks
```shell
memray flamegraph --leaks ~/tmp/memray/marie.bin.3658615
```

[](https://bloomberg.github.io/memray/run.html)
https://towardsdatascience.com/how-to-add-git-hooks-for-your-python-projects-using-the-pre-commit-framework-773acc3b28a7

https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates

https://flake8.pycqa.org/en/latest/