use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::model::{Kpis, OperationAttention};

pub(super) fn compact_identifier(value: &str, max_chars: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_owned();
    }
    if max_chars < 5 {
        return chars.into_iter().take(max_chars).collect();
    }
    let suffix_len = 4.min((max_chars - 1) / 2);
    let prefix_len = max_chars - suffix_len - 1;
    let prefix = chars.iter().take(prefix_len).collect::<String>();
    let suffix = chars
        .iter()
        .skip(chars.len() - suffix_len)
        .collect::<String>();
    format!("{prefix}…{suffix}")
}

pub(super) fn format_short_count(value: u64) -> String {
    match value {
        0..=999 => value.to_string(),
        1_000..=999_999 => format!("{}k", value / 1_000),
        1_000_000..=999_999_999 => format!("{}m", value / 1_000_000),
        _ => format!("{}b", value / 1_000_000_000),
    }
}

pub(super) fn json_scalar(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "N/A".to_owned(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        _ => "N/A".to_owned(),
    }
}

pub(super) fn timestamp_clock(timestamp: &str) -> String {
    DateTime::parse_from_rfc3339(timestamp).map_or_else(
        |_| "N/A".to_owned(),
        |value| value.format("%H:%M:%S").to_string(),
    )
}

pub(super) fn attention_label(attention: &[OperationAttention]) -> String {
    attention.first().map_or_else(
        || "--".to_owned(),
        |item| {
            item.count.map_or_else(
                || item.code.clone(),
                |count| format!("{} x{count}", item.code),
            )
        },
    )
}

pub(super) fn format_operation_age(seconds: f64) -> String {
    let seconds = seconds.max(0.0);
    if seconds < 1.0 {
        format!("{:.0}ms", seconds * 1_000.0)
    } else if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3_600.0 {
        format!("{:.0}m", seconds / 60.0)
    } else if seconds < 86_400.0 {
        format!("{:.1}h", seconds / 3_600.0)
    } else {
        format!("{:.1}d", seconds / 86_400.0)
    }
}

pub(super) fn compact_runtime_id(value: Option<&str>) -> String {
    let Some(value) = value else {
        return "N/A".to_owned();
    };
    let Some((host, id)) = value.split_once(':') else {
        return value.chars().take(18).collect();
    };
    format!("{host}:{}", id.chars().take(8).collect::<String>())
}

pub(super) fn format_optional_ratio(numerator: Option<u64>, denominator: Option<u64>) -> String {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) => format!("{numerator}/{denominator}"),
        _ => "N/A".to_owned(),
    }
}

pub(super) fn format_optional_triplet(
    first: Option<u64>,
    second: Option<u64>,
    third: Option<u64>,
) -> String {
    match (first, second, third) {
        (Some(first), Some(second), Some(third)) => format!("{first}/{second}/{third}"),
        _ => "N/A".to_owned(),
    }
}

pub(super) fn format_optional_ms(value: Option<f64>) -> String {
    value.map_or_else(|| "N/A".to_owned(), |value| format!("{value:.1}ms"))
}

pub(super) fn format_optional_number(value: Option<f64>) -> String {
    value.map_or_else(|| "N/A".to_owned(), |value| format!("{value:.2}"))
}

pub(super) fn format_seconds(value: Option<f64>) -> String {
    value.map_or_else(|| "N/A".to_owned(), format_operation_age)
}

pub(super) fn timeline_clock(value: &str) -> String {
    DateTime::parse_from_rfc3339(value).map_or_else(
        |_| "--:--".to_owned(),
        |value| value.format("%H:%M").to_string(),
    )
}

pub(super) fn compact_values(values: &[String]) -> String {
    match values {
        [] => "N/A".to_owned(),
        [value] => value.clone(),
        [first, rest @ ..] => format!("{first} +{}", rest.len()),
    }
}

pub(super) fn short_id(value: &str) -> String {
    value.chars().take(7).collect()
}

pub(super) fn compact_address(address: &str) -> String {
    let Some((host, port)) = address.rsplit_once(':') else {
        return address.to_owned();
    };
    let parts = host.rsplit('.').take(2).collect::<Vec<_>>();
    if parts.len() == 2 {
        format!("{}.{}:{port}", parts[1], parts[0])
    } else {
        address.to_owned()
    }
}

pub(super) fn bool_label(value: Option<bool>) -> String {
    value.map_or_else(
        || "N/A".to_owned(),
        |value| if value { "yes" } else { "no" }.to_owned(),
    )
}

pub(super) fn state_epoch(state: Option<&str>, epoch: Option<u64>) -> String {
    match (state, epoch) {
        (Some(state), Some(epoch)) => format!("{state}@{epoch}"),
        (Some(state), None) => state.to_owned(),
        _ => "N/A".to_owned(),
    }
}

pub(super) fn format_rate(value: Option<f64>) -> String {
    value.map_or_else(
        || "N/A".to_owned(),
        |rate| {
            if rate < 100.0 {
                format!("{rate:.1}/s")
            } else {
                format!("{rate:.0}/s")
            }
        },
    )
}

pub(super) fn format_percent(value: Option<f64>) -> String {
    value.map_or_else(|| "N/A".to_owned(), |percent| format!("{percent:.1}%"))
}

pub(super) fn rate_summary(kpis: &Kpis) -> Option<String> {
    let mut rates = Vec::new();
    if let Some(rate) = kpis.completed_plans_per_hour {
        rates.push(format!("plans {rate:.1}/h"));
    }
    if let Some(rate) = kpis.completed_executor_tasks_per_hour {
        rates.push(format!("exec tasks {rate:.1}/h"));
    }
    if let Some(success) = kpis.plan_success_rate_pct {
        rates.push(format!("plans ok {success:.1}%"));
    }
    if let Some(success) = kpis.task_success_rate_pct {
        rates.push(format!("tasks ok {success:.1}%"));
    }
    if kpis.gateway_requests_per_second.is_some() {
        rates.push(format!(
            "req {}",
            format_rate(kpis.gateway_requests_per_second)
        ));
    }
    if kpis.jobs_completed_per_second.is_some() {
        rates.push(format!(
            "done {}",
            format_rate(kpis.jobs_completed_per_second)
        ));
    }
    if kpis.job_failure_rate.is_some() {
        rates.push(format!("fail {}", format_percent(kpis.job_failure_rate)));
    }
    if let Some(p95) = kpis.gateway_p95_latency_ms {
        rates.push(format!("p95 {p95:.0}ms"));
    }
    (!rates.is_empty()).then(|| rates.join(" "))
}

pub(super) fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }
    formatted
}

pub(super) fn format_optional_count(value: Option<u64>) -> String {
    value.map_or_else(|| "N/A".to_owned(), format_count)
}

pub(super) fn format_optional_age(seconds: Option<f64>) -> String {
    seconds.map_or_else(
        || "N/A".to_owned(),
        |seconds| format_age(Duration::from_secs_f64(seconds.max(0.0))),
    )
}

pub(super) fn format_age(age: Duration) -> String {
    if age.as_secs() < 60 {
        format!("{}s", age.as_secs())
    } else {
        format!("{}m", age.as_secs() / 60)
    }
}

pub(super) fn timestamp_age(timestamp: Option<&str>) -> String {
    let Some(timestamp) = timestamp else {
        return "N/A".to_owned();
    };
    let Ok(observed_at) = DateTime::parse_from_rfc3339(timestamp) else {
        return "N/A".to_owned();
    };
    let seconds = Utc::now()
        .signed_duration_since(observed_at.with_timezone(&Utc))
        .num_seconds()
        .max(0) as u64;
    format_age(Duration::from_secs(seconds))
}
