from types import SimpleNamespace

from PIL import Image

from marie.engine.completion_contract import RequestContext
from marie.extract.annotators.llm_annotator import LLMAnnotator
from marie.extract.annotators.util import _build_request_contexts


def test_build_request_contexts_uses_source_identity_and_page_number_only():
    image = Image.new("RGB", (8, 8))
    batch_mapping = {
        0: (image, "prompt", "/tmp/frames/00001.png", ""),
        1: (image, "prompt", "/tmp/frames/frame_0002.png", "_t0"),
    }

    contexts = _build_request_contexts(
        batch_mapping,
        RequestContext(
            ref_id="PID_2_10832_0_255720425.tif",
            ref_type="stress",
            requested_pages=(0,),
        ),
    )

    assert contexts == [
        RequestContext(
            ref_id="PID_2_10832_0_255720425.tif",
            ref_type="stress",
            page_number=1,
            requested_pages=(0,),
        ),
        RequestContext(
            ref_id="PID_2_10832_0_255720425.tif",
            ref_type="stress",
            page_number=2,
            requested_pages=(0,),
        ),
    ]


def test_build_request_contexts_ignores_invalid_zero_page_number():
    image = Image.new("RGB", (8, 8))
    contexts = _build_request_contexts(
        {0: (image, "prompt", "/tmp/frames/frame_0000.png", "")},
        RequestContext(
            ref_id="PID_2_10832_0_255720425.tif",
            page_number=99,
        ),
    )

    assert contexts == [
        RequestContext(
            ref_id="PID_2_10832_0_255720425.tif",
            page_number=99,
        )
    ]


def test_llm_annotator_separates_span_metadata_from_request_context():
    annotator = object.__new__(LLMAnnotator)
    annotator.engine = SimpleNamespace(model_string="document-small")
    annotator.name = "mock_annotator_llm"
    annotator.layout_id = "layout-a"
    annotator.job_id = "job-runtime"
    annotator.dag_id = "dag-1"
    annotator.node_task_id = "node-1"
    annotator.llm_pool_id = "document-small"
    annotator.ref_id = "PID_2_10832_0_255720425.tif"
    annotator.ref_type = "stress"
    annotator.requested_pages = [0]

    metadata = annotator._build_span_metadata()
    request_context = annotator._build_model_request_context()

    assert "ref_id" not in metadata
    assert "ref_type" not in metadata
    assert "requested_pages" not in metadata
    assert metadata["job_id"] == "job-runtime"
    assert metadata["pool_id"] == "document-small"
    assert request_context == RequestContext(
        ref_id="PID_2_10832_0_255720425.tif",
        ref_type="stress",
        requested_pages=(0,),
    )


def test_model_request_context_preserves_none_requested_pages_as_all_pages():
    annotator = object.__new__(LLMAnnotator)
    annotator.ref_id = "PID_2_10832_0_255720425.tif"
    annotator.ref_type = "stress"
    annotator.requested_pages = None

    assert annotator._build_model_request_context() == RequestContext(
        ref_id="PID_2_10832_0_255720425.tif",
        ref_type="stress",
        requested_pages=None,
    )


def test_model_request_context_requires_source_identity():
    annotator = object.__new__(LLMAnnotator)
    annotator.ref_id = None
    annotator.ref_type = None
    annotator.requested_pages = [0]

    assert annotator._build_model_request_context() is None
