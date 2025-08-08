wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Generate a long input (≈2048 tokens).
-- This is just repeated text; real workloads may vary.
local long_input = string.rep("The quick brown fox jumps over the lazy dog. ", 200)

wrk.body = [[
{
  "model": "Qwen/QwQ-32B",
  "messages": [
    {"role": "user", "content": "]] .. long_input .. [["}
  ],
  "max_tokens": 1024
}
]]
