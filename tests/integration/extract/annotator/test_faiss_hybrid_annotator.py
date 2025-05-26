from typing import List

import pytest

from marie.extract.annotators.faiss_hybrid_annotator import FaissHybridAnnotator
from marie.extract.structures.line_with_meta import LineWithMeta
from marie.extract.structures.unstructured_document import UnstructuredDocument


@pytest.fixture
def dummy_ocr_lines():
    return [
        "TAX ID: 123456789    PATIENT ACCT: ABC123    CLAIM NUMBER: 987654321    PROVIDER NPI: 54321",
        "PATIENT NAME: JOHN DOE    MEMBER NUMBER: 100200300    CHECK NUMBER: 888777666    CHECK DATE: 01/01/2024",
        "PATIENT NAME JANE K SMITH  PATIENT ID 123456    MEMBER NUMBER: 200300400    CLAIM NUMBER: 87654 3210"
    ]


@pytest.fixture
def annotator_conf():
    
    return {
        'name': 'key-value-faiss-hybrid', # This is the key that will be saving the annotated data in
        'annotator_type': 'faiss-hybrid',
        'model_name': 'jinaai/jina-embeddings-v2-base-en',  # 'sentence-transformers/all-MiniLM-L6-v2'
        'top_k_candidates': 3,
        'fuzzy_threshold': 0.8,
        'embedding_threshold': 0.85,
        'fuzzy_weight': 0.3,
        'embedding_weight': 0.7,
        'min_final_score': 0.7,
        'min_acceptance_score': 0.7,
        'critical_fields': ['CLAIM NUMBER', 'PATIENT ACCT', 'CHECK NUMBER'],
        'critical_field_boost': 0.1,
        'memory_enabled': True,
        'memory_fields': [
            'CLAIM NUMBER', 'PATIENT ACCT', 'PATIENT NAME',
            'MEMBER NUMBER', 'CHECK NUMBER', 'CHECK DATE', 'PROVIDER NPI', 'PATIENT ID'
        ],
        'deduplicate_fields': False,
        'target_labels': [
            "TAX ID", "PATIENT ACCT", "PATIENT ID", "CLAIM NUMBER", "PROVIDER NPI",
            "PATIENT NAME", "MEMBER NUMBER", "CHECK NUMBER", "CHECK DATE"
        ],
        #
        'multiline_enabled': True,
        'multiline_threshold': 0.70,
        'multiline_window': 2,
        'multiline_reference_blocks': {
            'PatternPatient': [
                'TAX ID : 12345ABC PATIENT ACCT : 12345ABC CLAIM NUMBER : 12345ABC PROVIDER NPI : 12345ABC ',
                'PATIENT NAME : JOHN SMITH MEMBER NUMBER : 12345ABC CHECK NUMBER : 123456789 CHECK DATE : 04/01/2024'
            ]
        }
    }


@pytest.fixture
def layout_conf():
    return {
        'layout_id': 'dummy-layout',
        'source_filename': 'dummy.pdf'
    }


@pytest.fixture
def annotator_conf_yaml(annotator_conf, tmp_path):
    import yaml

    config_file = tmp_path / "annotator_config.yaml"
    yaml_content = yaml.dump(annotator_conf, default_flow_style=False)
    config_file.write_text(yaml_content)

    print('config_file path:', config_file)
    return yaml_content


@pytest.fixture
def annotator(tmp_path, annotator_conf, layout_conf):

    return FaissHybridAnnotator(str(tmp_path), annotator_conf, layout_conf)


def convert_to_lines_with_meta(ocr_lines: List[str]) -> List[LineWithMeta]:
    lines_with_meta = []
    for i, line in enumerate(ocr_lines):
        line_with_meta = LineWithMeta(
            line=line,
            metadata=None,
            annotations=[],
        )
        lines_with_meta.append(line_with_meta)
    return lines_with_meta


def doc_from_text_lines(ocr_lines: List[str]) -> UnstructuredDocument:
    lines_with_meta = convert_to_lines_with_meta(ocr_lines)
    return UnstructuredDocument(
        lines=lines_with_meta,
        metadata={
            "source_metadata": {
                "pages": 1,
            },
        },
        tables=[]
    )


def test_faiss_hybrid_annotator_run(dummy_ocr_lines, annotator, annotator_conf_yaml, tmp_path):
    print("\nAnnotator configuration in YAML:")
    print(annotator_conf_yaml)
    print(f"\nConfiguration saved to: {tmp_path / 'annotator_config.yaml'}")
    doc = doc_from_text_lines(dummy_ocr_lines)
    frames = []

    results = annotator.annotate(doc, frames)
    print('results')
    print(results)
    result = doc.annotations

    # Basic structure checks
    assert "document_metadata" in result
    assert isinstance(result["fields"], list)
    assert "errors" in result and isinstance(result["errors"], list)

    # Build a map of label -> list of extracted values
    extracted = {}
    for f in result["fields"]:
        extracted.setdefault(f["field_name"], []).append(f["value"])

    # Define exactly what we expect
    expected = {
        "TAX ID": ["123456789"],
        "PATIENT ACCT": ["ABC123"],
        "PROVIDER NPI": ["54321"],
        "CLAIM NUMBER": ["987654321", "87654 3210"],
        "PATIENT NAME": ["JOHN DOE", "JANE K SMITH"],
        "MEMBER NUMBER": ["100200300", "200300400"],
        "CHECK NUMBER": ["888777666"],
        "CHECK DATE": ["01/01/2024"],
        "PATIENT ID": ["123456"]
    }

    # Verify every expected label and its values
    for label, values in expected.items():
        assert label in extracted, f"Missing label: {label}"
        for v in values:
            assert v in extracted[label], f"Label {label} did not extract expected value '{v}'"

    # Ensure no unexpected fields creep in
    unexpected = set(extracted) - set(expected)
    assert not unexpected, f"Found unexpected fields: {unexpected}"

    # Validate confidences and that no empty strings slipped through
    for f in result["fields"]:
        assert f["confidence"] >= 0.6, f"Low confidence on {f['field_name']}"
        assert isinstance(f["value"], str) and f["value"].strip(), f"Empty value for {f['field_name']}"
