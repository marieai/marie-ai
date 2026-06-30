// HTTP Request Node for Marie Workflows
//
// A built-in node that makes HTTP requests using the host's HTTP client.
// This demonstrates how to create a reusable workflow node in Rust.

wit_bindgen::generate!({
    world: "node",
    path: "/opt/wit",
    generate_all,
});

use std::collections::HashMap;

// Configuration for HTTP requests (parsed from env.vars JSON)
#[derive(serde::Deserialize)]
struct HttpRequestConfig {
    method: String,
    url: String,
    #[serde(default)]
    headers: HashMap<String, String>,
    body: Option<String>,
    #[serde(default)]
    body_from_input: bool,
    auth_secret: Option<String>,
}

// Output structure for HTTP responses
#[derive(serde::Serialize)]
struct HttpResponseOutput {
    status: u16,
    headers: HashMap<String, String>,
    body: String,
    success: bool,
}

fn execute_http_request(input: Vec<Item>, env: Env, ctx: Context) -> Response {
    // Parse configuration from env.vars
    let config: HttpRequestConfig = match serde_json::from_str(&env.vars) {
        Ok(c) => c,
        Err(e) => {
            log_message(
                marie::node::console::LogLevel::Error,
                &format!("Failed to parse config: {}", e),
            );
            return Response::Err(format!("Invalid configuration: {}", e));
        }
    };

    // Log the request
    log_message(
        marie::node::console::LogLevel::Info,
        &format!(
            "HTTP Request: {} {} (node: {})",
            config.method, config.url, ctx.node_id
        ),
    );

    // Build headers from config
    let mut headers: Vec<(String, String)> = config
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Add Authorization header from secret if specified
    if let Some(auth_secret) = &config.auth_secret {
        if let Some(token) = marie::node::secrets::get(auth_secret) {
            headers.push(("Authorization".to_string(), format!("Bearer {}", token)));
        }
    }

    // Process each input item (or just make one request if no input)
    let items_to_process: Vec<Item> = if input.is_empty() {
        vec![Item {
            json: "{}".to_string(),
            binary: None,
        }]
    } else {
        input
    };

    let mut output_items: Vec<Item> = Vec::new();

    for item in items_to_process {
        // Prepare request body
        let body = if config.body_from_input {
            Some(item.json.as_bytes().to_vec())
        } else {
            config.body.as_ref().map(|b| b.as_bytes().to_vec())
        };

        // Make the HTTP request
        let request = marie::node::http_client::HttpRequest {
            method: config.method.clone(),
            url: config.url.clone(),
            headers: headers.clone(),
            body,
        };

        match marie::node::http_client::fetch(&request) {
            Ok(response) => {
                let response_headers: HashMap<String, String> =
                    response.headers.into_iter().collect();

                let output = HttpResponseOutput {
                    status: response.status,
                    headers: response_headers,
                    body: String::from_utf8_lossy(&response.body).to_string(),
                    success: response.status >= 200 && response.status < 300,
                };

                output_items.push(Item {
                    json: serde_json::to_string(&output).unwrap_or_else(|_| "{}".to_string()),
                    binary: Some(response.body),
                });
            }
            Err(e) => {
                log_message(
                    marie::node::console::LogLevel::Error,
                    &format!("HTTP request failed: {}", e),
                );

                let output = HttpResponseOutput {
                    status: 0,
                    headers: HashMap::new(),
                    body: e.clone(),
                    success: false,
                };

                output_items.push(Item {
                    json: serde_json::to_string(&output).unwrap_or_else(|_| "{}".to_string()),
                    binary: None,
                });
            }
        }
    }

    Response::Ok(output_items)
}

// Helper to log messages via the console interface
fn log_message(level: marie::node::console::LogLevel, message: &str) {
    marie::node::console::log(level, message);
}

// Component implementation
struct HttpRequestNode;

impl Guest for HttpRequestNode {
    fn execute(input: Vec<Item>, env: Env, ctx: Context) -> Response {
        execute_http_request(input, env, ctx)
    }
}

export!(HttpRequestNode);
