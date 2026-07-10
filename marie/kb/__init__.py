"""Knowledge-base gateway API extension (routes registered via
``_extend_rest_function``; execution lives in the KB executor)."""

from marie.kb.gateway_routes import register_kb_routes

__all__ = ["register_kb_routes"]
