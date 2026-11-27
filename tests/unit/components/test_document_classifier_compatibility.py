def test_classifier_default_and_legacy_import_are_page_level() -> None:
    from marie.components.document_classifier import TransformersDocumentClassifier
    from marie.components.document_classifier.transformers import (
        TransformersPageLevelClassifier,
    )
    from marie.pipe.components import CLASSIFIER_REGISTRY

    assert TransformersDocumentClassifier is TransformersPageLevelClassifier
    assert CLASSIFIER_REGISTRY['transformers'] is TransformersPageLevelClassifier


def test_document_batches_infer_max_pages() -> None:
    import torch

    from marie.components.document_classifier.transformers import collate_fn_inference

    page = {
        key: torch.ones(shape)
        for key, shape in {
            'input_ids': (4,),
            'attention_mask': (4,),
            'bbox': (4, 4),
            'pixel_values': (3, 8, 8),
        }.items()
    }
    batch = collate_fn_inference(
        [{'pages': [page]}, {'pages': [page, page]}], 'classifier', None
    )
    assert batch['page_mask'].tolist() == [[True, False], [True, True]]
