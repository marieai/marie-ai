"""Shared utilities for agent examples.

Common functions used across multiple agent example files.
"""

import os
from typing import Optional

from marie.agent import MarieEngineLLMWrapper, OpenAICompatibleWrapper

# Default models for each backend
DEFAULT_MODELS = {
    "marie": "qwen2_5_vl_7b",
    "openai": "gpt-4o-mini",
}


def create_llm(
    backend: str = "marie",
    model: Optional[str] = None,
):
    """Create an LLM wrapper based on the backend.

    Args:
        backend: "marie" or "openai"
        model: Model name (uses defaults if not specified)

    Returns:
        LLM wrapper instance (MarieEngineLLMWrapper or OpenAICompatibleWrapper)

    Raises:
        ValueError: If backend is unknown or required env vars are missing

    Environment Variables:
        OPENAI_API_KEY: Required for OpenAI backend
        OPENAI_API_URL: Optional, defaults to https://api.openai.com/v1
    """
    if backend == "marie":
        return MarieEngineLLMWrapper(engine_name=model or DEFAULT_MODELS["marie"])
    elif backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_URL")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable required for OpenAI backend"
            )
        return OpenAICompatibleWrapper(
            model=model or DEFAULT_MODELS["openai"],
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1/",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'marie' or 'openai'")


def print_debug_response(response, iteration: int = 0):
    """Print detailed debug information about an LLM response.

    Args:
        response: The response message (dict or Message object)
        iteration: Current iteration number for labeling
    """
    print(f"\n{'=' * 60}")
    print(f"[DEBUG] Response #{iteration}")
    print('=' * 60)

    if isinstance(response, dict):
        print(f"  Role: {response.get('role', 'N/A')}")
        content = response.get('content')
        print(
            f"  Content: {str(content)[:300]}{'...' if content and len(str(content)) > 300 else ''}"
        )

        if response.get('tool_calls'):
            print(f"  Tool Calls:")
            for tc in response['tool_calls']:
                if isinstance(tc, dict):
                    print(f"    - ID: {tc.get('id')}")
                    print(f"      Function: {tc.get('function', {}).get('name')}")
                    print(f"      Args: {tc.get('function', {}).get('arguments')}")
                else:
                    print(f"    - ID: {tc.id}")
                    print(f"      Function: {tc.function.name}")
                    print(f"      Args: {tc.function.arguments}")

        if response.get('tool_call_id'):
            print(f"  Tool Call ID: {response['tool_call_id']}")
    else:
        # Message object
        print(f"  Role: {getattr(response, 'role', 'N/A')}")
        content = getattr(response, 'content', None)
        print(
            f"  Content: {str(content)[:300]}{'...' if content and len(str(content)) > 300 else ''}"
        )

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"  Tool Calls:")
            for tc in response.tool_calls:
                print(f"    - ID: {tc.id}")
                print(f"      Function: {tc.function.name}")
                print(f"      Args: {tc.function.arguments}")

        if hasattr(response, 'tool_call_id') and response.tool_call_id:
            print(f"  Tool Call ID: {response.tool_call_id}")

    print('=' * 60)
