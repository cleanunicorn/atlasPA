"""
providers/__init__.py

Factory function: reads LLM_PROVIDER from environment and returns
the correct BaseLLMProvider instance.
"""

import os
from .base import BaseLLMProvider


def get_provider() -> BaseLLMProvider:
    """
    Return the configured LLM provider.

    Reads LLM_PROVIDER from environment (default: "anthropic").
    Supported values: "anthropic", "openai", "ollama", "openrouter"

    Wraps the provider with LoggingProvider unless LLM_LOG_FILE=off.
    """
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        inner = AnthropicProvider()
    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider

        inner = OpenAIProvider()
    elif provider_name == "ollama":
        from .ollama_provider import OllamaProvider

        inner = OllamaProvider()
    elif provider_name == "openrouter":
        from .openrouter_provider import OpenRouterProvider

        inner = OpenRouterProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            "Supported values: anthropic, openai, ollama, openrouter"
        )

    if os.getenv("LLM_LOG_FILE", "").lower() == "off":
        return inner

    from .logging_provider import LoggingProvider

    return LoggingProvider(inner)
