import tiktoken


def estimate_token_count(message: str, model_name: str = "gpt-3.5-turbo-0301") -> int:
    """
    Estimates the number of tokens in the given prompt using tiktoken.

    Args:
        message: The text input for which to estimate token count.
        model_name: The model name from OpenAI to select the tokenizer. Defaults to "gpt-3.5-turbo-0301".

    Returns:
        The estimated number of tokens in the prompt.
    """
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(message)
        return len(tokens)
    except Exception as e:
        raise ValueError(f"An error occurred while estimating tokens: {e}")
