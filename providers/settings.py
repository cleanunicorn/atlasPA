"""
Shared provider-level settings.
"""

import os


DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
