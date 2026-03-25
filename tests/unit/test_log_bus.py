import json
import logging
import queue
import threading
import time

from marie.logging_core.job_log_sink import JobLogSink
from marie.logging_core.log_bus import BatchingQueueListener


def _make_record(
    msg: str,
    request_id: str = "job-1",
    level: int = logging.INFO,
) -> logging.LogRecord:
    record = logging.LogRecord("test.logger", level, __file__, 123, msg, (), None)
    record.request_id = request_id
    record.context = "layer-main"
    return record


class _BlockingFlushSink(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._io_lock = threading.Lock()
        self.emit_started = threading.Event()
        self.release_emit = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        with self._io_lock:
            self.emit_started.set()
            self.release_emit.wait(timeout=2.0)

    def flush(self) -> None:
        with self._io_lock:
            pass


class _BatchAwareSink(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.batch_messages: list[list[str]] = []
        self.emit_calls = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.emit_calls += 1

    def handle_many(self, records: list[logging.LogRecord]) -> None:
        self.batch_messages.append([record.getMessage() for record in records])


class _RecordingHandle:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _RecordingJobLogSink(JobLogSink):
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir=log_dir, max_handles=10)
        self.recording_handles: dict[str, _RecordingHandle] = {}

    def _open_handle(self, file_path: str):
        handle = _RecordingHandle()
        self.recording_handles[file_path] = handle
        return handle


def test_listener_stop_timeout_does_not_block_on_slow_sink() -> None:
    listener = BatchingQueueListener(
        queue.Queue(),
        [_BlockingFlushSink()],
        batch_size=1,
        flush_interval=0.01,
    )
    sink = listener.handlers[0]
    assert isinstance(sink, _BlockingFlushSink)

    listener.start()
    listener.q.put(_make_record("slow write"))

    assert sink.emit_started.wait(timeout=1.0)
    worker = listener._thr
    assert worker is not None

    started = time.monotonic()
    listener.stop(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert worker.is_alive()
    assert listener._thr is worker

    sink.release_emit.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()

    listener.stop(timeout=0.5)
    assert listener._thr is None


def test_listener_prefers_handle_many_when_available() -> None:
    sink = _BatchAwareSink()
    listener = BatchingQueueListener(queue.Queue(), [sink], batch_size=8)

    listener._emit_to_handlers(
        [_make_record("one"), _make_record("two"), _make_record("three")],
        [sink],
    )

    assert sink.emit_calls == 0
    assert sink.batch_messages == [["one", "two", "three"]]


def test_job_log_sink_handle_many_batches_writes_per_request_id(tmp_path) -> None:
    sink = _RecordingJobLogSink(str(tmp_path))

    sink.handle_many(
        [
            _make_record("first", request_id="job-1"),
            _make_record("second", request_id="job-1"),
            _make_record("third", request_id="job-2"),
        ]
    )

    job1_path = sink.get_log_file_path("job-1")
    job2_path = sink.get_log_file_path("job-2")

    assert len(sink.recording_handles[job1_path].writes) == 1
    assert len(sink.recording_handles[job2_path].writes) == 1

    job1_entries = [
        json.loads(line)
        for line in sink.recording_handles[job1_path].writes[0].splitlines()
    ]
    job2_entries = [
        json.loads(line)
        for line in sink.recording_handles[job2_path].writes[0].splitlines()
    ]

    assert [entry["msg"] for entry in job1_entries] == ["first", "second"]
    assert [entry["msg"] for entry in job2_entries] == ["third"]
    sink.close()
