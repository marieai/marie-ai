"""marie/markitdown extension plugin — born-digital document extraction.

Speaks the marie-plugin-daemon stdio protocol (newline-delimited JSON events, a
heartbeat loop, and per-session request/response) exactly like the daemon's
fixture plugin. Exposes one tool action, ``convert``, which turns a local
document into Markdown via the ``markitdown`` library declared in
``requirements.txt`` (installed only into the plugin's own venv).
"""

import json
import os
import sys
import threading
import time

_emit_lock = threading.Lock()


def emit(payload):
    with _emit_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def heartbeat_loop():
    while True:
        emit({"session_id": "", "event": "heartbeat", "data": None})
        time.sleep(2)


def _load_markitdown():
    from markitdown import MarkItDown

    return MarkItDown()


def convert_document(path, fmt=None, *, converter_factory=None):
    """Convert a local document to Markdown.

    Returns ``{"markdown": str, "metadata": {...}}``. ``converter_factory`` is a
    seam for tests; it defaults to constructing a real ``MarkItDown`` and is only
    imported when actually converting so the module loads without the library.
    """
    if not path:
        raise ValueError("convert requires a 'path'")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such file: {path}")
    converter = (converter_factory or _load_markitdown)()
    result = converter.convert(path)
    markdown = _result_text(result)
    return {"markdown": markdown, "metadata": _build_metadata(path, fmt, result)}


def _result_text(result):
    for attr in ("text_content", "markdown"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value
    return str(result)


def _build_metadata(path, fmt, result):
    metadata = {}
    title = getattr(result, "title", None)
    if isinstance(title, str) and title.strip():
        metadata["title"] = title.strip()
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".")).lower()
    if fmt == "pdf":
        pages = _pdf_page_count(path)
        if pages is not None:
            metadata["page_count"] = pages
    return metadata


def _pdf_page_count(path):
    try:
        from pdfminer.pdfpage import PDFPage
    except ImportError:
        return None
    try:
        with open(path, "rb") as handle:
            return sum(1 for _ in PDFPage.get_pages(handle))
    except Exception:
        return None


def _extract_input(payload):
    """Pull the tool input out of the daemon payload.

    The daemon forwards the whole invoke payload opaquely, so tolerate either a
    ``tool_parameters`` wrapper or the input placed directly at the top level.
    """
    if not isinstance(payload, dict):
        return {}
    params = payload.get("tool_parameters")
    if isinstance(params, dict) and params:
        return params
    return payload


def _session_event(session_id, event_type, data):
    return {
        "session_id": session_id,
        "event": "session",
        "data": {"type": event_type, "data": data},
    }


def _log(level, message):
    return {
        "session_id": "",
        "event": "log",
        "data": {"level": level, "message": message},
    }


def dispatch_request(request, *, converter_factory=None):
    """Turn one inbound ``request`` event into the protocol events to emit back."""
    session_id = request.get("session_id")
    payload = request.get("data")
    params = _extract_input(payload)
    action = (
        payload.get("action", "convert") if isinstance(payload, dict) else "convert"
    )
    if action != "convert":
        return [
            _session_event(
                session_id, "error", {"message": f"unknown action: {action!r}"}
            )
        ]
    try:
        result = convert_document(
            params.get("path"),
            params.get("format"),
            converter_factory=converter_factory,
        )
    except Exception as error:
        return [_session_event(session_id, "error", {"message": str(error)})]
    return [
        _session_event(session_id, "stream", result),
        _session_event(session_id, "end", {}),
    ]


def main():
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit(_log("error", "invalid json line"))
            continue
        if not request.get("session_id"):
            emit(_log("error", "missing session_id"))
            continue
        if request.get("event") != "request":
            emit(_log("error", f"unsupported event: {request.get('event')!r}"))
            continue
        for message in dispatch_request(request):
            emit(message)


if __name__ == "__main__":
    main()
