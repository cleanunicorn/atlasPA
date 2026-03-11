"""
providers/openrouter_provider.py

LLM provider for OpenRouter — OpenAI-compatible API with access to many models.
"""

import os
import openai
from .openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider — routes to any supported model via OpenAI-compatible API.

    Inherits complete(), stream(), and _build_messages() from OpenAIProvider
    since the OpenRouter API is fully OpenAI-compatible.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )
        self._model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

    @property
    def model_name(self) -> str:
        return self._model
