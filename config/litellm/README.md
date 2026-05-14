

# LiteLLM

LiteLLM-Proxy is a lightweight proxy server for LLMs, designed to be easy to set up and use. It supports multiple LLMs and provides a simple API for interacting with them.

## Role In Marie LLM Dispatch

LiteLLM is the recommended provider-routing backend for Marie's LLM Dispatch Runtime when you need provider fallback, model routing, budgets, or rate limits.

Responsibility split:

- Marie Gateway / LLM Dispatch Runtime:
  - receives executor-originated LLM calls
  - queues requests in Valkey
  - tracks producer liveness, in-flight state, drops, retry/circuit-breaker state, and completed dispatch spans
  - calls one configured OpenAI-compatible backend URL
- LiteLLM:
  - receives that OpenAI-compatible call
  - applies provider fallback, deployment routing, budgets, rate limits, and provider-specific configuration
  - forwards the request to the final model backend

Point the gateway dispatch runtime at LiteLLM with:

```shell
OPENAI_API_BASE=http://litellm:4000/v1
OPENAI_API_KEY=<litellm-key>
```

Do not duplicate LiteLLM provider policy inside Marie Dispatch unless there is a Marie-specific workflow requirement that cannot be expressed in the provider gateway.

```shell
litellm --config ./config/litellm/config.yml --detailed_debug
litellm --config /mnt/data/marie-ai/config/litellm/config.yml --detailed_debug
```

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "mistral-small-latest",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```
