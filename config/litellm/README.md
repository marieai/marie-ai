

# LiteLLM

LiteLLM-Proxy is a lightweight proxy server for LLMs, designed to be easy to set up and use. It supports multiple LLMs and provides a simple API for interacting with them.


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

