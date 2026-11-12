import importlib

import pytest
from docarray import DocList

from marie.api import AssetKeyDoc


@pytest.mark.parametrize(
    'module,class_name,method',
    [
        (
            'marie.executor.classifier.document_classifier_executor',
            'DocumentClassificationExecutor',
            'classify',
        ),
        (
            'marie.executor.extract.document_serverless_executor',
            'DocumentServerlessExecutor',
            'evaluator',
        ),
        (
            'marie.executor.extract.document_annotator_executor',
            'DocumentAnnotatorExecutor',
            '_setup_request',
        ),
        (
            'marie.executor.template_matching.template_matching_executor',
            'TemplateMatchingExecutor',
            'match',
        ),
    ],
)
@pytest.mark.parametrize('count', [0, 2])
def test_invalid_document_count_returns_worker_failure(
    module, class_name, method, count
):
    executor_type = getattr(importlib.import_module(module), class_name)
    executor = object.__new__(executor_type)
    docs = DocList[AssetKeyDoc](
        [AssetKeyDoc(asset_key='s3://fixture/document.pdf') for _ in range(count)]
    )
    result = getattr(executor, method)(docs, {'job_id': 'invalid-input'})
    assert result['status'] == 'error'
    assert result['error'] == (
        'empty payload' if count == 0 else 'expected single document'
    )


@pytest.mark.parametrize('count', [0, 2])
def test_text_extractor_already_rejects_invalid_document_count(count):
    from marie.executor.text.text_extraction_executor import TextExtractionExecutor

    executor = object.__new__(TextExtractionExecutor)
    docs = DocList[AssetKeyDoc](
        [AssetKeyDoc(asset_key='s3://fixture/document.pdf') for _ in range(count)]
    )
    parameters = {
        'job_id': 'invalid-input',
        'ref_id': 'document',
        'ref_type': 'extract',
        'payload': {},
    }
    with pytest.raises(ValueError, match='Expected single document'):
        executor.extract(docs, parameters)
