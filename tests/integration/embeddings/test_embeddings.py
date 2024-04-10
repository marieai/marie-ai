import time

import pytest
from numpy.linalg import norm

from marie.embeddings.jina.jina_embeddings import JinaEmbeddings

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))


@pytest.mark.asyncio
async def test_jina_embeddings():
    jina_embeddings = JinaEmbeddings(
        model_name_or_path="hf://jinaai/jina-embeddings-v2-base-en"
    )
    start_o = time.time()
    for i in range(100):
        start = time.time()
        results = jina_embeddings.get_embeddings(
            ["How is the weather today?", "What is the current weather like today?"],
            truncation=True,
            max_length=512,
        )
        assert results is not None
        embeddings = results.embeddings
        assert results.embeddings.shape == (2, 768)
        assert cos_sim(embeddings[0], embeddings[1]) > 0.9
        val = cos_sim(embeddings[0], embeddings[1])
        print("Time taken: ", round(val, 4), time.time() - start)
    start_o = time.time() - start_o
    print("Total time taken: ", start_o)
