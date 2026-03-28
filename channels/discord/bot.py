"""
channels/discord/bot.py

Discord channel adapter.
Receives messages from Discord → forwards to Brain → sends response back.

Supported interactions:
  - Direct Messages (DMs) to the bot
  - @mentions in any server channel

Slash commands:
  /clear   — Reset conversation history for this user
  /status  — Show agent status

Security:
    Only responds to user IDs listed in DISCORD_ALLOWED_USERS.
    Set this in your .env to prevent others from using your agent.

Required .env:
    DISCORD_BOT_TOKEN=<your bot token>
    DISCORD_ALLOWED_USERS=123456789,987654321  (optional)
"""

import base64
import logging
import os
import time
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from memory.history import ConversationHistory
from paths import UPLOADS_DIR

logger = logging.getLogger(__name__)

# Files that can be sent as images in Discord embeds
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_UPLOAD_DIR = UPLOADS_DIR


class DiscordBot:
    """
    Discord channel adapter built on discord.py.

    Uses a Client (not Bot prefix-commands) with slash commands via app_commands.
    Responds to DMs and @mentions.
    """

    def __init__(self, brain):
        self.brain = brain
        self._history = ConversationHistory()

        token = os.getenv("DISCORD_BOT_TOKEN", "")
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN not set in environment")
        self._token = token

        allowed_raw = os.getenv("DISCORD_ALLOWED_USERS", "")
        self._allowed_users: set[int] = set()
        if allowed_raw.strip():
            for uid in allowed_raw.split(","):
                try:
                    self._allowed_users.add(int(uid.strip()))
                except ValueError:
                    logger.warning(f"Invalid user ID in DISCORD_ALLOWED_USERS: {uid}")

        if not self._allowed_users:
            logger.warning(
                "DISCORD_ALLOWED_USERS is empty — bot will respond to ANYONE. "
                "Set this to your Discord user ID for security."
            )

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)

        self._register_events()
        self._register_commands()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_allowed(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _user_id_str(self, user: discord.User | discord.Member) -> str:
        return f"discord_{user.id}"

    # ── Events ────────────────────────────────────────────────────────────────

    def _register_events(self) -> None:

        @self._client.event
        async def on_ready():
            await self._tree.sync()
            agent_name = os.getenv("AGENT_NAME", "Atlas")
            logger.info(
                f"Discord bot ready — logged in as {self._client.user} ({agent_name})"
            )

        @self._client.event
        async def on_message(message: discord.Message):
            # Ignore own messages
            if message.author == self._client.user:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mention = self._client.user in message.mentions

            if not is_dm and not is_mention:
                return

            if not self._is_allowed(message.author.id):
                await message.reply("⛔ Unauthorized.")
                return

            # Strip the @mention prefix if present
            content = message.content
            if is_mention and self._client.user:
                content = content.replace(f"<@{self._client.user.id}>", "").strip()
                content = content.replace(f"<@!{self._client.user.id}>", "").strip()

            user_id = self._user_id_str(message.author)

            # Check for audio attachments first
            audio_attachments = [
                a
                for a in message.attachments
                if a.content_type and a.content_type.startswith("audio/")
            ]
            if audio_attachments:
                await self._handle_audio(message, audio_attachments, content)
                return

            # Build multimodal content if image attachments are present
            image_attachments = [
                a
                for a in message.attachments
                if a.content_type and a.content_type.startswith("image/")
            ]
            if image_attachments:
                user_content: str | list = []
                if content:
                    user_content.append({"type": "text", "text": content})
                async with aiohttp.ClientSession() as session:
                    for att in image_attachments:
                        try:
                            async with session.get(att.url) as resp:
                                img_bytes = await resp.read()
                            media_type = att.content_type.split(";")[0]
                            img_b64 = base64.b64encode(img_bytes).decode()
                            user_content.append(
                                {
                                    "type": "image",
                                    "media_type": media_type,
                                    "data": img_b64,
                                }
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not download Discord attachment: {e}"
                            )
                if not user_content:
                    return
            else:
                if not content:
                    return
                user_content = content

            logger.info(f"Discord message from {message.author}: {str(content)[:100]}")

            async with message.channel.typing():
                history = self._history.load(user_id)
                placeholder = await message.reply("…")
                last_edit_at = 0.0

                async def _on_status(status: str) -> None:
                    nonlocal last_edit_at
                    now = time.monotonic()
                    if now - last_edit_at >= 1.0:
                        try:
                            await placeholder.edit(content=status)
                            last_edit_at = now
                        except Exception:
                            pass

                try:
                    response, updated_history = await self.brain.think(
                        user_message=user_content,
                        conversation_history=history,
                        on_status=_on_status,
                    )
                    self._history.save(user_id, updated_history)

                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    await _send_long(message.reply, response)

                    for path, caption in self.brain.take_files():
                        await _send_file(message.reply, Path(path), caption)

                except Exception as e:
                    logger.exception("Error in brain.think()")
                    try:
                        await placeholder.delete()
                    except Exception:
                        pass
                    await message.reply(f"⚠️ Something went wrong: {e}")

    # ── Audio handling ─────────────────────────────────────────────────────────

    async def _handle_audio(
        self,
        message: discord.Message,
        attachments: list,
        text: str,
    ) -> None:
        """Download audio attachment(s), transcribe via Parakeet, send to brain."""
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Download the first audio attachment
        att = attachments[0]
        filename = att.filename or f"audio_{att.id}"
        save_path = _UPLOAD_DIR / filename

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(att.url) as resp:
                    save_path.write_bytes(await resp.read())
            except Exception as e:
                await message.reply(f"⚠️ Could not download audio: {e}")
                return

        logger.info(f"Audio received from {message.author}: {filename} → {save_path}")

        # Transcribe
        transcript: str | None = None
        try:
            from channels.transcribe import transcribe

            transcript = await transcribe(save_path)
            logger.info(f"Transcribed ({filename}): {transcript[:120]}")
        except RuntimeError as e:
            logger.warning(f"Transcription unavailable: {e}")
        except Exception as e:
            logger.error(f"Transcription failed for {filename}: {e}")

        if transcript:
            user_message = transcript
            if text:
                user_message = f"{transcript}\n\n[User note: {text}]"
        else:
            user_message = (
                f"[System: the user sent a voice/audio message saved to {save_path}. "
                "Transcription is unavailable — nemo_toolkit may not be installed.]\n"
                f"{text or 'The user sent an audio message.'}"
            )

        user_id = self._user_id_str(message.author)
        async with message.channel.typing():
            history = self._history.load(user_id)
            try:
                response, updated_history = await self.brain.think(
                    user_message=user_message,
                    conversation_history=history,
                )
                self._history.save(user_id, updated_history)
                await _send_long(message.reply, response)
                for path, caption in self.brain.take_files():
                    await _send_file(message.reply, Path(path), caption)
            except Exception as e:
                logger.exception("Error processing audio message")
                await message.reply(f"⚠️ Something went wrong: {e}")

    # ── Slash commands ────────────────────────────────────────────────────────

    def _register_commands(self) -> None:

        @self._tree.command(name="clear", description="Reset your conversation history")
        async def cmd_clear(interaction: discord.Interaction):
            if not self._is_allowed(interaction.user.id):
                await interaction.response.send_message(
                    "⛔ Unauthorized.", ephemeral=True
                )
                return
            self._history.clear(self._user_id_str(interaction.user))
            await interaction.response.send_message(
                "🧹 Conversation cleared.", ephemeral=True
            )

        @self._tree.command(name="status", description="Show agent status")
        async def cmd_status(interaction: discord.Interaction):
            if not self._is_allowed(interaction.user.id):
                await interaction.response.send_message(
                    "⛔ Unauthorized.", ephemeral=True
                )
                return
            user_id = self._user_id_str(interaction.user)
            history = self._history.load(user_id)
            provider_name = self.brain.provider.model_name
            skills = self.brain.skills.all_skill_names()
            context_entries = len(self.brain.memory.parse_context_entries())
            await interaction.response.send_message(
                f"**Agent Status**\n"
                f"Model: `{provider_name}`\n"
                f"Conversation: {len(history)} messages\n"
                f"Long-term memories: {context_entries}\n"
                f"Skills: {', '.join(skills) or 'none'}",
                ephemeral=True,
            )

    # ── Push (heartbeat) ──────────────────────────────────────────────────────

    async def push_message(self, text: str, files: list | None = None) -> None:
        """
        Proactively send a message to all allowed users (called by the heartbeat).
        Sends as a DM to each allowed user.
        """
        if not self._allowed_users:
            logger.warning(
                "push_message: DISCORD_ALLOWED_USERS is empty — nowhere to push"
            )
            return

        for user_id in self._allowed_users:
            try:
                user = await self._client.fetch_user(user_id)
                dm = await user.create_dm()
                await _send_long(dm.send, text)
                for path, caption in files or []:
                    await _send_file(dm.send, Path(path), caption)
            except Exception as e:
                logger.error(f"push_message: failed to DM user {user_id}: {e}")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Discord and start processing events (non-blocking setup)."""
        logger.info("Starting Discord bot...")
        await self._client.login(self._token)
        import asyncio

        asyncio.get_event_loop().create_task(self._client.connect())

    async def stop(self) -> None:
        """Disconnect from Discord."""
        logger.info("Stopping Discord bot...")
        await self._client.close()


# ── Module-level helpers ───────────────────────────────────────────────────────


async def _send_long(send_fn, text: str) -> None:
    """Send text, splitting at 2000 chars (Discord limit)."""
    for i in range(0, max(1, len(text)), 2000):
        await send_fn(text[i : i + 2000])


async def _send_file(reply_fn, path: Path, caption: str) -> None:
    """Attach a file to a Discord message."""
    if not path.exists():
        await reply_fn(f"⚠️ File not found: {path.name}")
        return
    try:
        f = discord.File(str(path), filename=path.name)
        kwargs = {"file": f}
        if caption:
            kwargs["content"] = caption
        await reply_fn(**kwargs)
    except Exception as e:
        logger.error(f"Failed to send file {path.name}: {e}")
        await reply_fn(f"⚠️ Could not send file '{path.name}': {e}")
