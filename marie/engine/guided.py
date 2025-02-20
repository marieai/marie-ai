from enum import Enum


# https://docs.vllm.ai/en/latest/features/structured_outputs.html
class GuidedMode(str, Enum):
    CHOICE = "choice"
    REGEX = "regex"
    JSON = "json"
    GRAMMAR = "grammar"
