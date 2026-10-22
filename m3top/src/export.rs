use std::{
    fs,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ExportContext {
    pub gateway: String,
    pub fabric: String,
    pub job_id: String,
    pub history_id: u64,
}

pub fn write_failure_report(
    directory: &Path,
    context: &ExportContext,
    report: &Value,
) -> Result<PathBuf> {
    if report["job"]["job_id"].as_str() != Some(context.job_id.as_str())
        || report["selected_event"]["history_id"].as_u64() != Some(context.history_id)
        || !context
            .job_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        bail!("gateway report does not match the selected job and execution event");
    }
    let markdown = failure_markdown(context, report);
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(directory)
        .with_context(|| format!("could not create {}", directory.display()))?;
    let directory = directory.canonicalize()?;
    let path = directory.join(format!(
        "job-{}-event-{}-{}-{:08x}.md",
        context.job_id,
        context.history_id,
        Utc::now().format("%Y%m%dT%H%M%SZ"),
        fastrand::u32(..)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .with_context(|| format!("could not create {}", path.display()))?;
    let result = file
        .write_all(markdown.as_bytes())
        .and_then(|()| file.sync_all());
    if let Err(error) = result {
        let _ = fs::remove_file(&path);
        return Err(error).context("could not finish writing the diagnostic report");
    }
    Ok(path)
}

fn failure_markdown(context: &ExportContext, report: &Value) -> String {
    let mut output = format!(
        "# Job failure diagnostic report\n\nGateway: {}  \nFabric: {}  \nGenerated: {}\n\n",
        escape(&context.gateway),
        escape(&context.fabric),
        cell(&report["generated_at"])
    );
    output.push_str("The selected event is historical evidence. Current job state may reflect a later retry or completion. Times are recorded by the gateway and database.\n\n");
    output.push_str("## Selected execution event\n\n");
    fields(&mut output, &report["selected_event"]);
    if report["selected_event"]["error"]["traceback"]
        .as_str()
        .is_none_or(|text| text.is_empty())
    {
        output.push_str("No traceback was recorded for this event.\n\n");
    }
    output.push_str("## Selected execution attempt\n\n");
    fields(&mut output, &report["selected_attempt"]);
    output.push_str("## Current scheduler and worker state\n\n");
    fields(&mut output, &report["job"]);
    output.push_str("## Combined timeline\n\n| Time | Source | Status | Attempt | Message |\n| --- | --- | --- | --- | --- |\n");
    let mut events: Vec<(&str, &Value)> = Vec::new();
    for (section, source) in [
        ("worker_history", "worker"),
        ("scheduler_history", "scheduler"),
        ("attempts", "attempt"),
    ] {
        if let Some(rows) = report[section].as_array() {
            events.extend(rows.iter().map(|row| (source, row)));
        }
    }
    if !events.iter().any(|(source, row)| {
        *source == "worker" && row["history_id"].as_u64() == Some(context.history_id)
    }) {
        events.push(("worker", &report["selected_event"]));
    }
    let selected_attempt = &report["selected_attempt"];
    if selected_attempt.is_object()
        && !events.iter().any(|(source, row)| {
            *source == "attempt" && row["run_attempt_id"] == selected_attempt["run_attempt_id"]
        })
    {
        events.push(("attempt", selected_attempt));
    }
    events.sort_by_key(|(_, row)| {
        DateTime::parse_from_rfc3339(event_time(row))
            .map(|time| time.timestamp_millis())
            .unwrap_or(i64::MIN)
    });
    for (source, row) in events {
        let status = first(row, &["status", "terminal_status", "attempt_state"]);
        let fallback_message = first(
            row,
            &[
                "message",
                "worker_message",
                "terminal_reject_reason",
                "dispatch_error",
            ],
        );
        let message = if row["error"]["message"].is_null() {
            fallback_message
        } else {
            &row["error"]["message"]
        };
        output.push_str(&format!(
            "| {} | {source} | {} | {} | {} |\n",
            escape(event_time(row)),
            cell(status),
            cell(&row["run_attempt_id"]),
            cell(message)
        ));
    }
    output.push('\n');
    for (section, title) in [
        ("worker_history", "Worker execution history"),
        ("scheduler_history", "Scheduler state history"),
        ("attempts", "Durable execution attempts"),
    ] {
        output.push_str(&format!("## {title}\n\n"));
        match report[section].as_array() {
            Some(rows) if !rows.is_empty() => {
                for (index, row) in rows.iter().enumerate() {
                    output.push_str(&format!("### Record {}\n\n", index + 1));
                    fields(&mut output, row);
                }
            }
            _ => output.push_str("No retained records were returned.\n\n"),
        }
    }
    output.push_str("## Report scope and limits\n\n");
    output.push_str(&format!("History limit: {} newest retained records per source. The selected event and its attempt are fetched separately, even when older than the history window.\n\nText limit: {} bytes per field.\n\n", cell(&report["history_limit"]), cell(&report["text_limit"])));
    if let Some(limits) = report["truncated"].as_object() {
        for (section, value) in limits {
            output.push_str(&format!(
                "- {}: {}\n",
                escape(section),
                if value.as_bool() == Some(true) {
                    "truncated"
                } else {
                    "not truncated"
                }
            ));
        }
    }
    output.push_str("\nIncludes recorded error messages and tracebacks. Raw task payloads, output documents, runtime environment objects, and arbitrary attempt metadata are excluded. Common credential formats are redacted. Missing values mean unavailable.\n\n");
    output.push_str(&format!(
        "Diagnostic query reference: {}\n",
        cell(&report["source"])
    ));
    output
}

fn fields(output: &mut String, value: &Value) {
    let Some(object) = value.as_object() else {
        output.push_str("Unavailable.\n\n");
        return;
    };
    if object.is_empty() {
        output.push_str("No structured details were recorded.\n\n");
        return;
    }
    output.push_str("| Field | Value |\n| --- | --- |\n");
    for (key, value) in object {
        if !value.is_object()
            && !value.is_array()
            && key != "traceback"
            && key != "message"
            && key != "worker_message"
            && key != "scheduler_error_message"
        {
            output.push_str(&format!(
                "| {} | {} |\n",
                escape(&key.replace('_', " ")),
                cell(value)
            ));
        }
    }
    output.push('\n');
    for (key, value) in object {
        if value.is_object() {
            output.push_str(&format!("**{}**\n\n", escape(&key.replace('_', " "))));
            fields(output, value);
        } else if matches!(
            key.as_str(),
            "traceback" | "message" | "worker_message" | "scheduler_error_message"
        ) && !value.is_null()
        {
            output.push_str(&format!("**{}**\n\n", escape(&key.replace('_', " "))));
            let text = value
                .as_str()
                .filter(|text| !text.trim().is_empty())
                .unwrap_or("Not recorded");
            code_block(output, text);
        } else if value.is_array() {
            code_block(
                output,
                &serde_json::to_string_pretty(value).unwrap_or_default(),
            );
        }
    }
}

fn code_block(output: &mut String, text: &str) {
    let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    let fence = "`".repeat(3.max(longest + 1));
    output.push_str(&format!("{fence}text\n{text}\n{fence}\n\n"));
}

fn first<'a>(row: &'a Value, keys: &[&str]) -> &'a Value {
    keys.iter()
        .map(|key| &row[*key])
        .find(|value| !value.is_null())
        .unwrap_or(&Value::Null)
}

fn event_time(row: &Value) -> &str {
    first(row, &["changed_at", "terminal_at", "activated_at"])
        .as_str()
        .unwrap_or("Unavailable")
}

fn cell(value: &Value) -> String {
    match value {
        Value::Null => "Unavailable".into(),
        Value::String(text) if text.trim().is_empty() => "Not recorded".into(),
        Value::String(text) => escape(text),
        _ => escape(&value.to_string()),
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('|', "&#124;")
        .replace('`', "&#96;")
        .replace('\n', "<br>")
        .replace('\r', "")
}
