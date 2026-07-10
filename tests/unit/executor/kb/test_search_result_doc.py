from docarray import DocList

from marie.executor.kb.search_result_doc import SearchResultDoc


def test_search_result_doc_fields_survive_protobuf_round_trip():
    # VectorStoreExecutor's search/hybrid_search results cross the
    # worker -> gateway boundary as a protobuf-serialized DocList. A plain
    # TextDoc with attributes bolted on via `doc.metadata = {...}` loses
    # those attributes in that round trip because protobuf only encodes a
    # BaseDoc subclass's own declared fields. SearchResultDoc declares them,
    # so they must survive.
    docs = DocList[SearchResultDoc](
        [
            SearchResultDoc(
                id="node-1",
                text="hello world",
                similarity=0.87,
                text_score=0.42,
                rrf_score=0.05,
                source_id="src-1",
                node_type="chunk",
                index_name="kb-1",
                ref_doc_id="ref-1",
            )
        ]
    )

    restored = DocList[SearchResultDoc].from_protobuf(docs.to_protobuf())

    assert len(restored) == 1
    doc = restored[0]
    assert doc.id == "node-1"
    assert doc.text == "hello world"
    assert doc.similarity == 0.87
    assert doc.text_score == 0.42
    assert doc.rrf_score == 0.05
    assert doc.source_id == "src-1"
    assert doc.node_type == "chunk"
    assert doc.index_name == "kb-1"
    assert doc.ref_doc_id == "ref-1"


def test_search_result_doc_defaults():
    doc = SearchResultDoc(id="node-2", text="plain search result", similarity=0.5)

    assert doc.text_score is None
    assert doc.rrf_score is None
    assert doc.source_id == ""
    assert doc.node_type == ""
    assert doc.index_name == ""
    assert doc.ref_doc_id is None
