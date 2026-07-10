# marie/markitdown

Born-digital document extraction plugin for the marie-plugin-daemon. Converts
PDF-with-text-layer, DOCX, PPTX, XLSX, HTML, Markdown, and CSV files to Markdown
via [markitdown](https://github.com/microsoft/markitdown), bypassing the
rasterize + OCR path.

## Package layout

Mirrors the daemon's canonical plugin shape (`marie-extension.yaml` +
`main.py` + `requirements.txt`, zipped plain — no `.difypkg`):

- `marie-extension.yaml` — `kind: ExtensionPackage`, `runtime.type: python_source`,
  `entrypoint: main`.
- `main.py` — speaks the daemon stdio protocol (newline-JSON events, heartbeat
  loop, per-session request/response). One tool action, `convert`.
- `requirements.txt` — `markitdown` extras; installed into the plugin venv only.

## Tool: `convert`

Input `{path: str, format: str}` → output
`{markdown: str, metadata: {title?, page_count?}}`. `page_count` is filled for
PDFs via pdfminer (bundled with `markitdown[pdf]`).

## Packaging

```bash
./scripts/package.sh
```

Produces `marie-markitdown_<version>.zip` (version read from the manifest) in
the output directory (`/mnt/data/marie-ai/plugins/` by default). Install it via
the daemon's `POST /v1/plugins/install` with the raw zip bytes as the body.
