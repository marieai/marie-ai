import pytest
from numpy.linalg import norm

from marie.embeddings.jina.jina_embeddings import JinaEmbeddings

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))


@pytest.mark.asyncio
async def test_jina_embeddings():
    jina_embeddings = JinaEmbeddings(
        model_name_or_path="hf://jinaai/jina-embeddings-v2-base-en"
    )
    results = jina_embeddings.get_embeddings(
        ["How is the weather today?", "What is the current weather like today?"],
        truncation=True,
        max_length=2048,
    )
    assert results is not None
    embeddings = results.embeddings
    assert results.embeddings.shape == (2, 768)
    assert cos_sim(embeddings[0], embeddings[1]) > 0.9

    print(cos_sim(embeddings[0], embeddings[1]))
