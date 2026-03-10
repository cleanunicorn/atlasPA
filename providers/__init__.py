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
    Supported values: "anthropic", "openai", "ollama"
    """
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    elif provider_name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()
    elif provider_name == "openrouter":
        from .openrouter_provider import OpenRouterProvider
        return OpenRouterProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            "Supported values: anthropic, openai, ollama, openrouter"
        )
