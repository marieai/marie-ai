import os
import time

import pytest
import torch
from docarray import DocList
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from marie.api.docs import StorageDoc
from marie.embeddings.sentence_transformers.sentence_transformers_embeddings import (
    SentenceTransformerEmbeddings,
)
from marie.executor.storage.PostgreSQLStorage import PostgreSQLStorage
from marie.logging_core.profile import TimeContext

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, "docker-compose.yml")

print(compose_yml)


def _tags(
    index: int, ftype: str, checksum: str, embeddings: str, embedding_size: int
) -> dict:
    return {
        "action": "classifier",
        "index": index,
        "type": ftype,
        "ttl": 48 * 60,
        "checksum": checksum,
        "runtime": {"version": "0.0.1", "type": "python"},
        "embeddings": embeddings,
        "embedding_size": embedding_size,
    }


@pytest.fixture()
def docker_compose(request):
    os.system(
        f"docker compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker compose -f {request.param} --project-directory . down --remove-orphans"
    )


#  docker-compose -f docker-compose.yml --project-directory . up  --build  --remove-orphans
@pytest.mark.parametrize("docker_compose", [compose_yml], indirect=["docker_compose"])
def test_storage(tmpdir, docker_compose):
    # def test_storage(tmpdir):
    # benchmark only
    nr_docs = 1000

    storage = PostgreSQLStorage()
    handler = storage.handler

    with TimeContext(f"### rolling insert {nr_docs} docs"):
        print("Testing insert")

        ref_id = "test"
        docs = DocList[StorageDoc](
            [
                StorageDoc(
                    content={"test": "Test", "xyz": "Greg"},
                    tags=_tags(index, "metadata", ref_id, "none", 0),
                )
                for index in range(nr_docs)
            ]
        )

        handler.clear()

        handler.add(docs, **{"ref_id": ref_id, "ref_type": "test"})
        size = handler.get_size()

        assert nr_docs == size


# @pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_embedding_storage(tmpdir):
    # def test_storage(tmpdir):
    # benchmark only
    nr_docs = 1
    storage = PostgreSQLStorage()
    handler = storage.handler
    handler.clear()

    # create two embeddings for testing
    texts = ["The dog is barking", "The cat is purring", "The bear is growling"]

    provider = SentenceTransformerEmbeddings(
        devices=["cpu"], use_gpu=False, batch_size=1, show_error=True
    )
    embeddings = provider.get_embeddings_raw(texts)

    with TimeContext(f"### rolling insert {nr_docs} docs"):
        print("Testing insert")
        docs = DocList[StorageDoc]()

        for text, embedding in zip(texts, embeddings):
            print("Text: ", text)
            doc = StorageDoc(
                embedding=embedding,
                tags=_tags(0, "metadata", "test", "sentence-transformers", 768),
            )
            docs.append(doc)

        storage.add(
            docs,
            "embedding",
            {"ref_id": "test", "ref_type": "test"},
        )

        cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

        # for each text, get the embedding and compare it to the original
        for text, embedding in zip(texts, embeddings):
            print("Text: ", text)
            results = storage.similarity_search_with_score(embedding, 1)
            result_embedding = results[0][1]
            sim_score = cos_sim(embedding, result_embedding)
            print("Results: ", sim_score)
            assert sim_score == 1.0  # the similarity score should be 1.0
