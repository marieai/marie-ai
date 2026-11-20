use serde_json::{Map, Value};

pub(super) fn deployment_entries(
    data: Option<&Value>,
    field: &str,
) -> Vec<(String, Map<String, Value>)> {
    data.and_then(|value| value.get(field))
        .and_then(Value::as_object)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|(key, value)| {
                    value.as_object().map(|entry| (key.clone(), entry.clone()))
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn frontier_summary(data: Option<&Value>) -> Option<&Value> {
    data.and_then(|value| {
        value
            .get("frontier")
            .and_then(|frontier| frontier.get("summary"))
            .or_else(|| value.get("frontier_summary"))
    })
}

pub(super) fn state_rank(state: &str) -> u8 {
    match state {
        "unavailable" => 0,
        "degraded" => 1,
        "saturated" => 2,
        "running" => 3,
        "idle" => 4,
        _ => 5,
    }
}

pub(super) fn text(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

pub(super) fn optional_text(value: Option<&Value>) -> Option<String> {
    text(value).map(str::to_owned)
}

pub(super) fn optional_integer(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

pub(super) fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

pub(super) fn integer(value: Option<&Value>) -> u64 {
    value.and_then(Value::as_u64).unwrap_or(0)
}
