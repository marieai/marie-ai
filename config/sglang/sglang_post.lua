wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = [[
{
  "model": "Qwen/QwQ-32B",
  "messages": [
    {"role": "user", "content": "Write a short poem about the ocean."}
  ],
  "max_tokens": 128
}
]]
