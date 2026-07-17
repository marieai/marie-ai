from types import SimpleNamespace

from marie.extract.annotators.util import _prompt_lines_by_page


def _page(text: str) -> dict:
    width = len(text) * 8 + 16
    return {
        'words': [
            {
                'id': 0,
                'text': text,
                'box': [8, 8, len(text) * 8, 20],
                'line': 0,
                'word_index': 0,
            }
        ],
        'lines': [
            {
                'line': 0,
                'wordids': [0],
                'text': text,
                'bbox': [8, 8, len(text) * 8, 20],
                'confidence': 1.0,
            }
        ],
        'meta': {
            'page': 0,
            'imageSize': {'width': width, 'height': 32},
        },
    }


def test_semantic_document_prompt_text_preserves_markdown_lines() -> None:
    markdown_row = '| Subject | Estimate | Margin of Error | Percent |'
    document = SimpleNamespace(
        source_metadata={
            'extraction': {
                'result_kind': 'semantic_document',
                'ocr_invoked': False,
            },
            'ocr': [_page(markdown_row)],
        }
    )

    lines = _prompt_lines_by_page(document)

    assert lines[0][0]['text'] == markdown_row

