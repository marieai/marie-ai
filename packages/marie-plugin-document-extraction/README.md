# Document Extraction Plugin

Combined MarkItDown, Docling Slim, and Tree-sitter extraction providers for Marie.

Document formats (PDF, Office, OpenDocument, web, email, EPUB, LaTeX) produce semantic Markdown or structured JSON. Source-code formats — 40 languages driven by the vendored tags queries in `marie_plugins/document_extraction/queries/` — produce a symbol outline (`markdown`), a structured symbol table (`json`, `schemas/code-symbols-v1.json`) with AST spans for knowledge-base chunking, the syntax tree (`cst` — a `.txt` artifact, one pre-order line per node: `row:col-row:col type`, leaf text in backticks; named nodes only by default, nesting recoverable from span containment), or a queryable node table (`nodes` — a `.jsonl` artifact, one JSON object per node with `id`/`parent`/`type`/`start`/`end`/`bytes`/leaf `text`, made for SQL over code structure without any parsing: `clickhouse-client -q "INSERT INTO code_nodes FORMAT JSONEachRow" < document.jsonl`, or `duckdb -c "SELECT type, count() FROM read_json('document.jsonl') GROUP BY type"`). `provider_options` accepts `include_references` (reference capture sites), `include_cst`, `include_markdown`, and `include_anonymous` (adds punctuation/keyword tokens to `cst`/`nodes`, matching `tree-sitter parse --cst`). With the include flags enabled, one `json` artifact carries the symbol table, references, CST, and markdown outline together. A language is advertised only when its grammar loads and its query compiles (probed at runtime); adding one is a `<language>-tags.scm` file plus an extension mapping, and rich fixtures back the most-used languages. Repo-level ingestion — cloning, tree walking, ranking — is the caller's concern; this plugin parses one file per request.

This is a first-party Marie system plugin maintained under `packages/`. Its
`marie-plugin-` directory and distribution prefix distinguish executable
plugins from shared libraries while its stable extension ID remains
`ext.marie.document-extraction`.

The plugin contributes `marie_plugins.document_extraction` to the implicit
`marie_plugins` namespace. It intentionally does not contain
`marie_plugins/__init__.py` or `marie_plugins.runtime`; the daemon supplies the
runtime namespace portion.

The plugin runs in the uv environment created by `marie-plugin-daemon`. The
daemon injects its stdlib-only `marie_plugins.runtime` for session framing and
heartbeats. The plugin does not import Marie core or `marie-extension`. The
stdio protocol carries bounded capability and artifact descriptors; extracted
document bodies are written to the request output directory.

## Actions

- `capabilities`: returns the provider-format edges ready in the installed lock.
- `extract`: detects the input, dispatches providers, and writes a result artifact.

## Invoking Directly

Without arguments, `main.py` serves daemon requests over stdin/stdout — that is the production mode and takes no command-line parameters; every request arrives as one JSON line. With arguments, it runs one request through the same handler and prints the response frames, which is the quickest way to test or debug the plugin by hand:

```bash
uv sync --locked

# one-shot debug mode
uv run --locked python main.py capabilities
uv run --locked python main.py extract tests/fixtures/sample.pdf /tmp/out
uv run --locked python main.py extract tests/fixtures/sample.csv /tmp/out --provider markitdown
uv run --locked python main.py extract tests/fixtures/sample.csv /tmp/out --provider docling --no-fallback
uv run --locked python main.py extract tests/fixtures/sample.csv /tmp/out --option table_mode=accurate --option max_pages=5
uv run --locked python main.py extract tests/fixtures/sample.pptx /tmp/out --output-format html
uv run --locked python main.py --help
```

The exit code is 0 on success and 1 when any frame is a typed error, so one-shot runs are scriptable.

To run without `uv`, use the plugin venv's interpreter directly — the system `python3` has none of the plugin's dependencies. The runtime namespace must be importable: either prefix `PYTHONPATH`, or install it editable once (see `../marie-plugin-daemon/python_runtime/README.md`):

```bash
PYTHONPATH=../marie-plugin-daemon/python_runtime .venv/bin/python main.py extract tests/fixtures/sample.pdf /tmp/out

# or, after the one-time editable install into this venv:
uv pip install --no-deps -e ../marie-plugin-daemon/python_runtime
.venv/bin/python main.py extract tests/fixtures/sample.pdf /tmp/out
```

Build the production artifact with:

```bash
./scripts/package.sh /path/to/output
```

## Tests

The suite covers three layers of the same behavior: provider adapters in-process (`provider_cases.py`), the exposed handler functions directly for breakpoint debugging (`test_direct_invocation.py`), and the real command over the stdio protocol (`test_stdio_pdf_process.py`, `packaged_protocol.py`).

```bash
uv run --locked pytest -q                              # everything
uv run --locked pytest tests/test_direct_invocation.py -q   # in-process, debuggable
uv run --locked pytest tests/test_stdio_pdf_process.py -q   # real command over stdio
```

With the `marie-ai-pytorch-2-12` environment active, run the complete system
invocation from the Marie AI repository root with:

```bash
pytest tests/integration/plugins/test_document_extraction_embedded.py -q
```

This builds the current daemon, packages the plugin, invokes capabilities
through `EmbeddedPlugins`, verifies the 53 ready formats, and extracts the HTML
fixture through the same system entrypoint.
