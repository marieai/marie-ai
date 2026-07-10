import pytest

from marie.executor.kb.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, chunk_overlap=2) == []


def test_text_shorter_than_chunk_size_returns_single_chunk():
    chunks = chunk_text("hello", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert (chunks[0].start, chunks[0].end) == (0, 5)


def test_text_exactly_chunk_size_returns_single_chunk():
    text = "a" * 10
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_no_overlap_produces_contiguous_chunks():
    text = "0123456789"
    chunks = chunk_text(text, chunk_size=4, chunk_overlap=0)
    assert [c.text for c in chunks] == ["0123", "4567", "89"]
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (4, 8), (8, 10)]


def test_overlap_repeats_trailing_characters_in_next_chunk():
    text = "0123456789"
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=2)
    # step = 3; last window is clamped to text_len and the loop stops there
    assert [(c.start, c.end) for c in chunks] == [(0, 5), (3, 8), (6, 10)]
    assert chunks[0].text == "01234"
    assert chunks[1].text == "34567"
    assert chunks[1].text[:2] == chunks[0].text[-2:]


def test_last_chunk_does_not_exceed_text_length_or_duplicate():
    text = "x" * 11
    chunks = chunk_text(text, chunk_size=5, chunk_overlap=1)
    assert chunks[-1].end == len(text)
    # no chunk starts past the end of the text
    assert all(c.start < len(text) for c in chunks)
    # reconstructing without overlap covers the whole text with no gaps
    covered_end = 0
    for c in chunks:
        assert c.start <= covered_end
        covered_end = max(covered_end, c.end)
    assert covered_end == len(text)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_raises(chunk_size):
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=chunk_size, chunk_overlap=0)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=-1)


def test_overlap_greater_or_equal_to_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=5, chunk_overlap=5)
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=5, chunk_overlap=6)
