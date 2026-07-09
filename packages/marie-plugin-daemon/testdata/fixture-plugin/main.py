"""Fixture plugin speaking the dify_plugin stdio protocol with zero dependencies."""

import json
import sys
import threading
import time

_emit_lock = threading.Lock()
_storage_sessions = {}


def emit(payload):
    with _emit_lock:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()


def heartbeat_loop():
    while True:
        emit({"session_id": "", "event": "heartbeat", "data": None})
        time.sleep(2)


def emit_storage_invoke(session_id, request_id, request):
    emit(
        {
            "session_id": session_id,
            "event": "session",
            "data": {
                "type": "invoke",
                "data": {
                    "backwards_request_id": request_id,
                    "type": "storage",
                    "request": request,
                },
            },
        }
    )


def handle_storage_response(session_id, state, inner):
    event = inner.get("event")
    if event == "response":
        state["stash"] = inner.get("data")
        return
    if event == "error":
        state["error"] = inner.get("message")
        return
    if event != "end":
        return
    del _storage_sessions[session_id]
    if "error" in state:
        emit(
            {
                "session_id": session_id,
                "event": "session",
                "data": {
                    "type": "stream",
                    "data": {"storage_error": state["error"], "step": state["step"]},
                },
            }
        )
        emit(
            {
                "session_id": session_id,
                "event": "session",
                "data": {"type": "end", "data": {}},
            }
        )
        return
    if state["step"] == "set":
        state["step"] = "get"
        _storage_sessions[session_id] = state
        emit_storage_invoke(session_id, "rt-get", {"opt": "get", "key": state["key"]})
        return
    data_hex = (state.get("stash") or {}).get("data", "")
    emit(
        {
            "session_id": session_id,
            "event": "session",
            "data": {
                "type": "stream",
                "data": {"storage_get": bytes.fromhex(data_hex).decode()},
            },
        }
    )
    emit(
        {
            "session_id": session_id,
            "event": "session",
            "data": {"type": "end", "data": {}},
        }
    )


def main():
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit(
                {
                    "session_id": "",
                    "event": "log",
                    "data": {"level": "error", "message": "invalid json line"},
                }
            )
            continue
        session_id = request.get("session_id")
        if not session_id:
            emit(
                {
                    "session_id": "",
                    "event": "log",
                    "data": {"level": "error", "message": "missing session_id"},
                }
            )
            continue
        event = request.get("event")
        if event == "backwards_response":
            state = _storage_sessions.get(session_id)
            if state is not None:
                handle_storage_response(session_id, state, request.get("data", {}))
                continue
            emit(
                {
                    "session_id": session_id,
                    "event": "session",
                    "data": {
                        "type": "stream",
                        "data": {"backwards": request.get("data", {})},
                    },
                }
            )
            emit(
                {
                    "session_id": session_id,
                    "event": "session",
                    "data": {"type": "end", "data": {}},
                }
            )
            continue
        if event != "request":
            emit(
                {
                    "session_id": "",
                    "event": "log",
                    "data": {"level": "error", "message": f"unknown event: {event!r}"},
                }
            )
            continue
        missing = [
            k
            for k in (
                "conversation_id",
                "message_id",
                "app_id",
                "endpoint_id",
                "context",
            )
            if k not in request
        ]
        if missing:
            emit(
                {
                    "session_id": session_id,
                    "event": "session",
                    "data": {
                        "type": "error",
                        "data": {"message": f"missing envelope fields: {missing}"},
                    },
                }
            )
            continue
        payload = request.get("data", {})
        if isinstance(payload, dict) and payload.get("emit_invoke"):
            emit(
                {
                    "session_id": session_id,
                    "event": "session",
                    "data": {
                        "type": "invoke",
                        "data": {"backwards_request_id": "r1", "type": "tool"},
                    },
                }
            )
            continue
        if isinstance(payload, dict) and payload.get("storage_roundtrip"):
            key = payload.get("key", "")
            value = payload.get("value", "")
            _storage_sessions[session_id] = {"step": "set", "key": key}
            emit_storage_invoke(
                session_id,
                "rt-set",
                {"opt": "set", "key": key, "value": value.encode().hex()},
            )
            continue
        if isinstance(payload, dict) and payload.get("storage_roundtrip_get_only"):
            key = payload.get("key", "")
            _storage_sessions[session_id] = {"step": "get", "key": key}
            emit_storage_invoke(session_id, "rt-get", {"opt": "get", "key": key})
            continue
        emit(
            {
                "session_id": session_id,
                "event": "session",
                "data": {"type": "stream", "data": {"echo": payload, "event": event}},
            }
        )
        emit(
            {
                "session_id": session_id,
                "event": "session",
                "data": {"type": "end", "data": {}},
            }
        )


if __name__ == "__main__":
    main()
