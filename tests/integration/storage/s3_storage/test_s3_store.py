import os
import time

import pytest

from marie.common.file_io import get_file_count
from marie.storage import StorageManager, S3StorageHandler
from marie.utils.utils import ensure_exists

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, "docker-compose.yml")


def is_running(container_name):
    """
    verify the status of a sniffer container by it's name
    :param container_name: the name of the container
    :return: Boolean if the status is ok
    """
    import docker

    DOCKER_CLIENT = docker.DockerClient(base_url="unix://var/run/docker.sock")
    RUNNING = "running"
    container = DOCKER_CLIENT.containers.get(container_name)
    container_state = container.attrs["State"]
    container_is_running = container_state["Status"] == RUNNING

    return container_is_running


@pytest.fixture()
def docker_compose(request):
    """
    Start the docker compose or skip the test if the container is already running
    in the background on the host.

    :param request:
    :return:
    """

    if is_running("s3server"):
        yield
    else:
        os.system(
            f"docker compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
        )
        time.sleep(10)
        yield
        os.system(
            f"docker compose -f {request.param} --project-directory . down --remove-orphans"
        )


def setup_storage():
    """Setup the storage handler"""
    handler = S3StorageHandler(
        config={
            "S3_ACCESS_KEY_ID": "MARIEACCESSKEY",
            "S3_SECRET_ACCESS_KEY": "MARIESECRETACCESSKEY",
            "S3_STORAGE_BUCKET_NAME": "marie",
            "S3_ENDPOINT_URL": "http://localhost:8000",
            "S3_ADDRESSING_STYLE": "path",
        }
    )

    # export AWS_ACCESS_KEY_ID=MARIEACCESSKEY; export AWS_SECRET_ACCESS_KEY=MARIESECRETACCESSKEY;  aws s3 ls --endpoint-url http://localhost:8000
    StorageManager.register_handler(handler=handler)
    StorageManager.ensure_connection("s3://")


@pytest.mark.parametrize("docker_compose", [compose_yml], indirect=["docker_compose"])
def test_storage_dir(tmpdir, docker_compose):
    setup_storage()

    # ensure nothing is present
    items = StorageManager.list("s3://marie")
    exists = StorageManager.exists("s3://marie")

    assert len(items) == 0
    assert not exists

    # validate that creating a directory works
    StorageManager.mkdir("s3://marie")
    exists = StorageManager.exists("s3://marie")
    assert exists

    # create number of  temp files and upload it to s3
    N = 100

    for i in range(N):
        temp_file = tmpdir.join(f"file_{i}.txt")
        temp_file.write("hello world")
        StorageManager.copy(str(temp_file), f"s3://marie/pid/file_{i}.txt")

    items = StorageManager.list("s3://marie/pid")
    assert len(items) == N

    items = StorageManager.list("s3://marie/pid/0")
    assert len(items) == 0


@pytest.mark.parametrize("docker_compose", [compose_yml], indirect=["docker_compose"])
def test_read_ops(tmpdir, docker_compose):
    setup_storage()

    StorageManager.mkdir("s3://marie")
    exists = StorageManager.exists("s3://marie")
    assert exists == True

    # Local file to remote and back
    temp_file = tmpdir.join(f"file.txt")
    temp_file.write("hello world")

    StorageManager.copy(str(temp_file), f"s3://marie/file.txt")

    # data = StorageManager.read(f"s3://marie/file.txt")
    temp_file_out = tmpdir.join(f"file-out.txt")
    StorageManager.copy(f"s3://marie/file.txt", str(temp_file_out))

    assert temp_file_out.read() == temp_file.read()


@pytest.mark.parametrize("docker_compose", [compose_yml], indirect=["docker_compose"])
def test_write_ops(tmpdir, docker_compose):
    setup_storage()

    StorageManager.mkdir("s3://marie")

    # Local file to remote and back
    temp_file = tmpdir.join(f"file.txt")
    temp_file.write("hello world")
    StorageManager.write(
        temp_file,
        f"s3://marie/file.txt",
    )

    # Read remote file to a byte array
    temp_file_out = tmpdir.join(f"file-out.txt")
    data = StorageManager.read(f"s3://marie/file.txt", overwrite=True)
    temp_file_out.write(data)

    assert temp_file_out.read() == temp_file.read()

    # Read remote file to a byte array
    temp_file_out = tmpdir.join(f"file-out.txt")
    StorageManager.read_to_file(f"s3://marie/file.txt", temp_file_out, overwrite=True)

    assert temp_file_out.read() == temp_file.read()


@pytest.mark.parametrize("docker_compose", [compose_yml], indirect=["docker_compose"])
def test_write_dir(tmpdir, docker_compose):
    setup_storage()
    StorageManager.ensure_connection()

    StorageManager.mkdir("s3://marie")

    dirs = []
    src_dir = tmpdir.mkdir("src")
    dst_dir = tmpdir.mkdir("dst")

    dirs.append(ensure_exists(os.path.join(src_dir, "a", "1")))
    dirs.append(ensure_exists(os.path.join(src_dir, "a", "1", "2")))
    dirs.append(ensure_exists(os.path.join(src_dir, "a", "1", "3")))

    dirs.append(ensure_exists(os.path.join(src_dir, "a", "2")))
    dirs.append(ensure_exists(os.path.join(src_dir, "b", "1")))
    dirs.append(ensure_exists(os.path.join(src_dir, "b", "2")))

    files = []
    N = 10

    for tmp_dir in dirs:
        for i in range(N):
            temp_file = os.path.join(tmp_dir, f"file_{i}.txt")
            with open(temp_file, "w") as f:
                f.write(f"hello marie")
            files.append(temp_file)

    # copy the files to s3
    StorageManager.copy_dir(
        src_dir, "s3://marie/wt", relative_to_dir=src_dir, match_wildcard="*"
    )

    expected = N * len(dirs)

    # list the files
    items = StorageManager.list("s3://marie/wt/")
    assert len(items) == expected

    # # copy the files back
    StorageManager.copy_remote("s3://marie/wt", dst_dir, match_wildcard="*")

    # check the files
    c1 = get_file_count(dst_dir)
    c2 = get_file_count(src_dir)

    assert c1 == c2
