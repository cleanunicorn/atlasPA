"""
providers/ollama_provider.py

LLM provider for Ollama (local models via OpenAI-compatible API).
Ollama exposes an OpenAI-compatible endpoint at http://localhost:11434/v1
"""

import os
from .openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """
    Ollama provider — wraps OpenAIProvider with Ollama's local endpoint.

    Ollama runs locally and exposes an OpenAI-compatible API, so we reuse
    OpenAIProvider and just point it at the local server.

    Set OLLAMA_MODEL in .env to choose the local model (default: llama3.2).
    """

    def __init__(self):
        # Set env vars that OpenAIProvider reads
        os.environ.setdefault(
            "OPENAI_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        os.environ.setdefault(
            "OPENAI_API_KEY", "ollama"
        )  # Ollama ignores the key but openai client requires one
        self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        super().__init__()
        self._model = self._ollama_model  # Override model set by parent

    @property
    def model_name(self) -> str:
        return f"ollama/{self._ollama_model}"
