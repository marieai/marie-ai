"""Exception hierarchy for the prompt template system."""


class PromptTemplateError(Exception):
    """Base exception for all prompt template errors."""


class PromptRenderError(PromptTemplateError):
    """Raised when template rendering fails."""


class PromptLoadError(PromptTemplateError):
    """Raised when a prompt file cannot be loaded from disk."""
