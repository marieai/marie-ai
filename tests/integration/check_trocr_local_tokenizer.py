"""Run manually with local TrOCR weights and assets/gpt2 tokenizer files."""

from pathlib import Path
from unittest.mock import patch

import requests
import torch

from marie.constants import __model_path__
from marie.document.trocr_ocr_processor import init


def main() -> None:
    model_dir = Path(__model_path__)
    tokenizer_dir = model_dir / 'assets' / 'gpt2'
    for filename in ('encoder.json', 'vocab.bpe'):
        if not (tokenizer_dir / filename).is_file():
            raise FileNotFoundError(tokenizer_dir / filename)

    request = requests.sessions.Session.request

    def block_tokenizer_download(
        session: requests.Session, method: str, url: str, **kwargs
    ) -> requests.Response:
        if 'gpt2_bpe' in url or 'huggingface.co' in url:
            raise AssertionError(f'Unexpected tokenizer request: {url}')
        kwargs.setdefault('timeout', 15)
        return request(session, method, url, **kwargs)

    torch.set_num_threads(4)
    with patch('requests.sessions.Session.request', block_tokenizer_download):
        _, cfg, task, _, bpe, _, _ = init(
            str(model_dir / 'trocr' / 'trocr-large-printed.pt'), beam=3, device='cpu'
        )
        assert cfg.bpe.gpt2_encoder_json == str(tokenizer_dir / 'encoder.json')
        assert cfg.bpe.gpt2_vocab_bpe == str(tokenizer_dir / 'vocab.bpe')
        for text in ('Hello world!', 'Invoice 12345: $67.89', 'Résumé — café'):
            assert bpe.decode(bpe.encode(text)) == text
            assert task.bpe.decode(task.bpe.encode(text)) == text


if __name__ == '__main__':
    main()
