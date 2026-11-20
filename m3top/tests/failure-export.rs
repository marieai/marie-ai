use m3top::export::{ExportContext, write_failure_report};
use serde_json::json;

#[test]
fn report_preserves_failure_evidence_and_writes_private_unique_markdown_files() {
    let directory = std::env::temp_dir().join(format!("m3top-export-{}", fastrand::u64(..)));
    let context = ExportContext {
        gateway: "test-gateway".into(),
        fabric: "default".into(),
        job_id: "job-a".into(),
        history_id: 912,
    };
    let report = json!({
        "generated_at": "2026-09-30T17:00:00Z", "history_limit": 250,
        "job": {"job_id": "job-a", "scheduler_state": "completed", "retry_count": 1},
        "selected_event": {"history_id": 912, "status": "FAILED", "run_attempt_id": "attempt-a",
            "changed_at": "2026-09-30T16:00:00Z", "executor_host": "192.168.106.75",
            "error": {"type": "QueueTaskError", "message": "failed | detail <script>",
                      "traceback": "Traceback\n```\nworker.py:412\ncall_timeout"}},
        "worker_history": [], "scheduler_history": [{"changed_at": "2026-09-30T16:01:00Z", "status": "retry"}],
        "attempts": [{"run_attempt_id": "attempt-a", "terminal_accepted": false}],
        "truncated": {"worker_history": true, "text": false}
    });
    let first = write_failure_report(&directory, &context, &report).unwrap();
    let second = write_failure_report(&directory, &context, &report).unwrap();
    assert_ne!(first, second);
    assert!(first.is_absolute());
    assert_eq!(first.extension().unwrap(), "md");
    let markdown = std::fs::read_to_string(&first).unwrap();
    for expected in [
        "test-gateway",
        "attempt-a",
        "192.168.106.75",
        "QueueTaskError",
        "worker.py:412",
        "call_timeout",
        "completed",
        "false",
        "truncated",
    ] {
        assert!(markdown.contains(expected), "missing {expected}");
    }
    assert!(markdown.contains("text\nfailed | detail <script>\n```"));
    assert!(markdown.contains("````text\nTraceback"));
    let timeline = markdown
        .split("## Combined timeline")
        .nth(1)
        .unwrap()
        .split("## Worker execution history")
        .next()
        .unwrap();
    assert!(timeline.find("16:00:00Z").unwrap() < timeline.find("16:01:00Z").unwrap());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(first).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn report_identity_mismatch_does_not_create_a_file() {
    let directory = std::env::temp_dir().join(format!("m3top-export-{}", fastrand::u64(..)));
    let context = ExportContext {
        gateway: "test".into(),
        fabric: "default".into(),
        job_id: "job-a".into(),
        history_id: 912,
    };
    let wrong = json!({"job": {"job_id": "job-b"}, "selected_event": {"history_id": 913}});
    assert!(write_failure_report(&directory, &context, &wrong).is_err());
    assert!(!directory.exists());
}
