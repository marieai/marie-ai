import json
from pathlib import Path

import pytest
import torch
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)

from marie.embeddings.openai.openai_trans_embeddings import (
    OpenAITransformerEmbeddings,
)


@pytest.mark.parametrize('load_through_registry', [True, False])
def test_loads_custom_clip_checkpoint_from_resolved_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    load_through_registry: bool,
) -> None:
    model_dir = tmp_path / 'relocated model zoo' / 'custom-clip'
    model_dir.mkdir(parents=True)
    if load_through_registry:
        (model_dir / 'marie.json').write_text(
            json.dumps({'_name_or_path': 'custom-clip'}), encoding='utf-8'
        )
    config = CLIPConfig(
        text_config={
            'vocab_size': 2,
            'hidden_size': 8,
            'intermediate_size': 16,
            'num_hidden_layers': 1,
            'num_attention_heads': 2,
            'max_position_embeddings': 8,
            'bos_token_id': 0,
            'eos_token_id': 1,
        },
        vision_config={
            'hidden_size': 8,
            'intermediate_size': 16,
            'num_hidden_layers': 1,
            'num_attention_heads': 2,
            'image_size': 8,
            'patch_size': 4,
        },
        projection_dim=4,
    )
    trained_model = CLIPModel(config)
    with torch.no_grad():
        trained_model.text_projection.weight.fill_(0.375)
    torch.save(
        {'model_state_dict': trained_model.state_dict()},
        model_dir / 'pytorch_model.bin',
    )
    (model_dir / 'vocab.json').write_text(
        json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1}), encoding='utf-8'
    )
    (model_dir / 'merges.txt').write_text('#version: 0.2\n', encoding='utf-8')
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    CLIPProcessor(
        image_processor=CLIPImageProcessor(), tokenizer=tokenizer
    ).save_pretrained(model_dir)
    monkeypatch.setattr(
        CLIPModel, 'from_pretrained', lambda *args, **kwargs: CLIPModel(config)
    )

    if load_through_registry:
        embeddings = OpenAITransformerEmbeddings(
            model_name_or_path=str(model_dir), use_gpu=False
        )
        model, processor, loaded_tokenizer = (
            embeddings.model,
            embeddings.processor,
            embeddings.tokenizer,
        )
    else:
        embeddings = OpenAITransformerEmbeddings.__new__(OpenAITransformerEmbeddings)
        model, processor, loaded_tokenizer = embeddings.setup_model(str(model_dir), 'cpu')

    torch.testing.assert_close(
        model.text_projection.weight,
        torch.full((4, 8), 0.375),
    )
    assert isinstance(processor, CLIPProcessor)
    assert isinstance(loaded_tokenizer, CLIPTokenizer)
