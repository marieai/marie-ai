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


import json

import pytest


@pytest.mark.parametrize(
    "marie_metadata", [None, {"architectures": ["LayoutLMv3ForSequenceClassification"]}]
)
def test_standard_hf_directory_uses_auto_model(tmp_path, monkeypatch, marie_metadata):
    from marie.components.document_classifier import transformers as component

    config = {
        "_name_or_path": "local",
        "model_type": "layoutlmv3",
        "architectures": ["LayoutLMv3ForSequenceClassification"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    if marie_metadata is not None:
        (tmp_path / "marie.json").write_text(
            json.dumps({**marie_metadata, "_name_or_path": "local"})
        )

    class ModelLoadRequested(Exception):
        pass

    def load(path):
        assert path == str(tmp_path)
        raise ModelLoadRequested

    monkeypatch.setattr(
        component.AutoModelForSequenceClassification, "from_pretrained", load
    )
    with pytest.raises(ModelLoadRequested):
        component.TransformersPageLevelClassifier(str(tmp_path), use_gpu=False)


@pytest.mark.parametrize(
    ("classifier_name", "architecture"),
    [
        ("TransformersDocumentLevelClassifier", "LayoutLMv3DocumentClassifier"),
        ("TransformersSplittingClassifier", "LayoutLMv3DocumentSplitter"),
    ],
)
def test_custom_classifier_uses_architecture_in_hf_config(
    tmp_path, monkeypatch, classifier_name, architecture
):
    from marie.components.document_classifier import transformers as component

    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "local", "architectures": [architecture]})
    )
    (tmp_path / "marie.json").write_text(json.dumps({"_name_or_path": "local"}))

    class ModelLoadRequested(Exception):
        pass

    def load(path):
        assert path == str(tmp_path)
        raise ModelLoadRequested

    monkeypatch.setattr(getattr(component, architecture), "from_pretrained", load)
    with pytest.raises(ModelLoadRequested):
        getattr(component, classifier_name)(str(tmp_path), use_gpu=False)
