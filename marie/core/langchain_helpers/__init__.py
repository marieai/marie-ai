"""Init file for langchain helpers."""

try:
    import langchain  # noqa  # pants: no-infer-dep
except ImportError:
    raise ImportError(
        "langchain not installed. "
        "Please install langchain with `uv add llama_index[langchain]`."
    )
