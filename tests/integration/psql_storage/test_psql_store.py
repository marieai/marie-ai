import collections
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from marie import Document, DocumentArray, Executor, Flow, requests
from marie.logging.profile import TimeContext
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, 'docker-compose.yml')

print(compose_yml)

@pytest.fixture()
def docker_compose(request):
    os.system(
        f'docker-compose -f {request.param} --project-directory . up  --build -d '
        f'--remove-orphans'
    )
    time.sleep(5)
    yield
    os.system(
        f'docker-compose -f {request.param} --project-directory . down '
        f'--remove-orphans'
    )

#  docker-compose -f docker-compose.yml --project-directory . up  --build  --remove-orphans
# @pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
# def test_storage(tmpdir, docker_compose):
def test_storage(tmpdir):
    # benchmark only
    nr_docs = 1000000

    storage = PostgreSQLStorage()
    handler = storage.handler
    print(storage)
    print(handler)

    with TimeContext(f'### rolling insert {nr_docs} docs'):
        print("Testing insert")

    payload = {
            "test":"Test",
            "xyz":"Greg"
    }

    dd = DocumentArray([Document(id=str(_), content=payload) for _ in range(10)])

    print(dd[2].content)
    handler.add(dd)

    # handler.add([payload])


    assert 1 == 1
