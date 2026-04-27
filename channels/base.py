"""
channels/base.py

Abstract base class for all Atlas channel adapters.

Every channel must implement:
    start()         — begin processing incoming events
    stop()          — gracefully shut down
    push_message()  — proactively send a message (called by the heartbeat)

Channels that enforce an allowlist of user IDs should call
_parse_allowed_users() from __init__ and use _is_allowed() at each entry point.
"""

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    """Shared interface and ACL helpers for all channel adapters."""

    def __init__(self) -> None:
        self._allowed_users: set[int] = set()

    def _parse_allowed_users(self, env_var: str, channel_name: str) -> None:
        """Populate _allowed_users from a comma-separated env var."""
        raw = os.getenv(env_var, "")
        if raw.strip():
            for uid in raw.split(","):
                try:
                    self._allowed_users.add(int(uid.strip()))
                except ValueError:
                    logger.warning(f"Invalid user ID in {env_var}: {uid}")
        if not self._allowed_users:
            logger.warning(
                f"{env_var} is empty — {channel_name} bot will respond to ANYONE. "
                f"Set this to your {channel_name} user ID for security."
            )

    def _is_allowed(self, user_id: int) -> bool:
        """Return True when user_id is permitted (or no allowlist is configured)."""
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def push_message(self, text: str, files: list | None = None) -> None: ...
