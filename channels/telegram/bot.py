"""
channels/telegram/bot.py

Telegram channel adapter.
Receives messages from Telegram → forwards to Brain → sends response back.

Security:
    Only responds to user IDs listed in TELEGRAM_ALLOWED_USERS.
    Set this in your .env to prevent others from using your agent.
"""

import base64
import logging
import os
import time
from pathlib import Path
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from memory.history import ConversationHistory
from channels.base import BaseChannel
from channels.telegram.formatting import md_to_html
from paths import UPLOADS_DIR

# Minimum seconds between Telegram message edits (rate limit: ~20 edits/min/chat)
_STREAM_EDIT_INTERVAL = 0.6

# Where incoming files are staged before the brain decides what to do with them
_UPLOAD_DIR = UPLOADS_DIR

logger = logging.getLogger(__name__)

# Persistent reply keyboard shown to all authorised users
_MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton("/clear"), KeyboardButton("/status"), KeyboardButton("/jobs")]],
    resize_keyboard=True,
    is_persistent=True,
)


class TelegramBot(BaseChannel):
    def __init__(self, brain, on_message_callback=None):
        """
        Args:
            brain:                  The Brain instance.
            on_message_callback:    Optional async callback for routing (used by Gateway).
        """
        super().__init__()
        self.brain = brain
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

        self._history = ConversationHistory()
        self._parse_allowed_users("TELEGRAM_ALLOWED_USERS", "Telegram")

        self.app = Application.builder().token(self.token).build()
        self._register_handlers()

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("jobs", self._cmd_jobs))
        self.app.add_handler(
            CallbackQueryHandler(self._handle_job_button, pattern=r"^run_job:")
        )
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self.app.add_handler(
            MessageHandler(filters.Document.ALL, self._handle_document)
        )
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self.app.add_handler(MessageHandler(filters.AUDIO, self._handle_voice))
        self.app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        agent_name = os.getenv("AGENT_NAME", "Atlas")
        history_len = len(self._history.load(str(user.id)))
        resume_note = (
            f" (resuming — {history_len} messages in history)" if history_len else ""
        )
        await update.message.reply_text(
            f"👋 Hi, I'm {agent_name}! Your personal AI agent.{resume_note}\n\n"
            f"Just send me a message to get started.",
            reply_markup=_MAIN_KEYBOARD,
        )

    async def _cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        self._history.clear(str(user_id))
        self.brain.reset_session_tokens()
        await update.message.reply_text(
            "🧹 Conversation cleared.", reply_markup=_MAIN_KEYBOARD
        )

    async def _reply(self, update: Update, text: str, reply_markup=None) -> None:
        """Send a reply with Telegram HTML formatting, splitting if needed."""
        html_text = md_to_html(text) or "✅ Done."
        if len(html_text) > 4096:
            # Split on paragraph boundaries when possible
            chunks = [html_text[i : i + 4096] for i in range(0, len(html_text), 4096)]
            for i, chunk in enumerate(chunks):
                markup = reply_markup if i == len(chunks) - 1 else None
                await update.message.reply_text(
                    chunk, parse_mode="HTML", reply_markup=markup
                )
        else:
            await update.message.reply_text(
                html_text, parse_mode="HTML", reply_markup=reply_markup
            )

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        history = self._history.load(str(user_id))
        provider_name = self.brain.provider.model_name
        skills = self.brain.skills.all_skill_names()
        context_entries = len(self.brain.memory.parse_context_entries())
        tokens = self.brain.session_tokens
        await self._reply(
            update,
            f"🤖 **Agent Status**\n"
            f"Model: `{provider_name}`\n"
            f"Conversation: {len(history)} messages\n"
            f"Long-term memories: {context_entries}\n"
            f"Skills: {', '.join(skills) or 'none'}\n"
            f"Tokens (session): {tokens['input']} in / {tokens['output']} out",
            reply_markup=_MAIN_KEYBOARD,
        )

    async def _cmd_jobs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        from heartbeat.jobs import load_jobs

        jobs = [j for j in load_jobs() if j.enabled]
        if not jobs:
            await update.message.reply_text(
                "No scheduled jobs configured.", reply_markup=_MAIN_KEYBOARD
            )
            return
        buttons = [
            [
                InlineKeyboardButton(
                    f"▶ {j.id}  ({j.schedule})", callback_data=f"run_job:{j.id}"
                )
            ]
            for j in jobs
        ]
        await update.message.reply_text(
            "Tap a job to run it now:",
            reply_markup=InlineKeyboardMarkup(buttons),
        )

    async def _handle_job_button(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        if not self._is_allowed(query.from_user.id):
            await query.answer("⛔ Unauthorized.")
            return
        job_id = query.data.removeprefix("run_job:")
        await query.answer(f"Triggering {job_id}…")

        heartbeat = getattr(self.brain, "heartbeat", None)
        if not heartbeat:
            await query.edit_message_text("⚠️ Heartbeat scheduler is not running.")
            return
        triggered = heartbeat.trigger_job(job_id)
        if triggered:
            await query.edit_message_text(
                f"▶ Job <b>{job_id}</b> is running — result will arrive shortly.",
                parse_mode="HTML",
            )
        else:
            await query.edit_message_text(f"⚠️ Job '{job_id}' not found.")

    async def _handle_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """
        Handle an incoming file/document from the user.

        Downloads the file to ~/agent-files/uploads/, then calls brain.think()
        with a system note describing what arrived and where it was saved.
        The brain can then decide what to do with it (e.g. call setup_account
        on a Google credentials JSON, or read a text file, etc.).
        """
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        doc = update.message.document
        filename = doc.file_name or f"upload_{doc.file_unique_id}"
        caption = update.message.caption or ""

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = _UPLOAD_DIR / filename

        await update.message.chat.send_action("typing")
        try:
            tg_file = await doc.get_file()
            await tg_file.download_to_drive(str(save_path))
        except Exception as e:
            await update.message.reply_text(f"⚠️ Could not download file: {e}")
            return

        logger.info(
            f"File received from {user.username or user.id}: {filename} → {save_path}"
        )

        # Build a context message for the brain
        reply_ctx = self._build_reply_context(update)
        user_message = (
            f"{reply_ctx}"
            f"[System: the user sent a file named '{filename}' "
            f"which has been saved to {save_path}]\n"
            f"{caption if caption else 'Please help me set up or use this file.'}"
        )

        user_id = str(user.id)
        history = self._history.load(user_id)
        try:
            response_text, updated_history = await self.brain.think(
                user_message=user_message,
                conversation_history=history,
            )
            self._history.save(user_id, updated_history)

            await self._reply(update, response_text, reply_markup=_MAIN_KEYBOARD)

            for path, caption_f in self.brain.take_files():
                await self._send_file(update, path, caption_f)

        except Exception as e:
            logger.exception("Error processing uploaded file")
            await update.message.reply_text(f"⚠️ Something went wrong: {e}")

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle an incoming voice note or audio file.

        Downloads the audio, transcribes it with NVIDIA Parakeet, then feeds
        the transcript to the brain exactly like a typed message.

        If Parakeet is not installed the brain is notified about the audio file
        instead so it can at least acknowledge the message.
        """
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        # Telegram Voice = recorded voice note (.ogg/Opus)
        # Telegram Audio = music/audio file sent as audio
        audio_obj = update.message.voice or update.message.audio
        if not audio_obj:
            return

        if update.message.voice:
            filename = f"voice_{audio_obj.file_unique_id}.ogg"
        else:
            filename = (
                getattr(audio_obj, "file_name", None)
                or f"audio_{audio_obj.file_unique_id}.ogg"
            )

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = _UPLOAD_DIR / filename

        await update.message.chat.send_action("typing")

        try:
            tg_file = await audio_obj.get_file()
            await tg_file.download_to_drive(str(save_path))
        except Exception as e:
            await update.message.reply_text(f"⚠️ Could not download audio: {e}")
            return

        logger.info(
            f"Audio received from {user.username or user.id}: {filename} → {save_path}"
        )

        # Transcribe with Parakeet
        caption = update.message.caption or ""
        transcript: str | None = None
        try:
            from channels.transcribe import transcribe

            await update.message.chat.send_action("typing")
            transcript = await transcribe(save_path)
            logger.info(f"Transcribed ({filename}): {transcript[:120]}")
        except RuntimeError as e:
            logger.warning(f"Transcription unavailable: {e}")
        except Exception as e:
            logger.error(f"Transcription failed for {filename}: {e}")

        reply_ctx = self._build_reply_context(update)
        if transcript:
            user_message = f"{reply_ctx}{transcript}"
            if caption:
                user_message = f"{reply_ctx}{transcript}\n\n[User note: {caption}]"
        else:
            user_message = (
                f"{reply_ctx}"
                f"[System: the user sent a voice/audio message saved to {save_path}. "
                "Transcription is unavailable — nemo_toolkit may not be installed.]\n"
                f"{caption or 'The user sent an audio message.'}"
            )

        user_id = str(user.id)
        history = self._history.load(user_id)
        try:
            response_text, updated_history = await self.brain.think(
                user_message=user_message,
                conversation_history=history,
            )
            self._history.save(user_id, updated_history)

            await self._reply(update, response_text, reply_markup=_MAIN_KEYBOARD)

            for path, cap in self.brain.take_files():
                await self._send_file(update, path, cap)

        except Exception as e:
            logger.exception("Error processing voice message")
            await update.message.reply_text(f"⚠️ Something went wrong: {e}")

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle an incoming photo from the user.

        Downloads the highest-resolution version, base64-encodes it, and
        builds a multimodal content list so the vision-capable LLM can see it.
        """
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        # Telegram sends multiple resolutions; take the largest
        photo = update.message.photo[-1]
        caption = update.message.caption or ""

        await update.message.chat.send_action("typing")
        try:
            tg_file = await photo.get_file()
            image_bytes = bytes(await tg_file.download_as_bytearray())
        except Exception as e:
            await update.message.reply_text(f"⚠️ Could not download image: {e}")
            return

        image_data = base64.b64encode(image_bytes).decode()

        # Build multimodal content: optional reply context + text caption + image block
        content: list = []
        reply_ctx = self._build_reply_context(update)
        if reply_ctx:
            content.append({"type": "text", "text": reply_ctx.rstrip()})
        if caption:
            content.append({"type": "text", "text": caption})
        content.append(
            {"type": "image", "media_type": "image/jpeg", "data": image_data}
        )

        user_id = str(user.id)
        history = self._history.load(user_id)
        try:
            response_text, updated_history = await self._stream_think(
                update, content, history
            )
            self._history.save(user_id, updated_history)

            for path, cap in self.brain.take_files():
                await self._send_file(update, path, cap)

        except Exception as e:
            logger.exception("Error processing photo")
            await update.message.reply_text(f"⚠️ Something went wrong: {e}")

    @staticmethod
    def _build_reply_context(update: Update) -> str:
        """
        If the incoming message is a reply to another message, return a string
        describing the original message so the brain has context.

        Returns an empty string when the message is not a reply.
        """
        replied = update.message.reply_to_message
        if not replied:
            return ""

        # Prefer plain text, then caption (media messages), then media type label
        if replied.text:
            quoted = replied.text
        elif replied.caption:
            quoted = replied.caption
        elif replied.photo:
            quoted = "[photo]"
        elif replied.document:
            name = getattr(replied.document, "file_name", None) or "file"
            quoted = f"[document: {name}]"
        elif replied.voice or replied.audio:
            quoted = "[voice/audio message]"
        elif replied.sticker:
            quoted = f"[sticker: {replied.sticker.emoji or ''}]"
        else:
            quoted = "[message]"

        # Truncate very long quoted messages to keep the prompt readable
        if len(quoted) > 300:
            quoted = quoted[:297] + "…"

        sender = ""
        if replied.from_user:
            sender = replied.from_user.first_name or replied.from_user.username or ""

        if sender:
            return f'[Replying to {sender}: "{quoted}"]\n'
        return f'[Replying to: "{quoted}"]\n'

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        reply_ctx = self._build_reply_context(update)
        user_text = (
            (reply_ctx + update.message.text) if reply_ctx else update.message.text
        )
        user_id = str(user.id)

        logger.info(f"Message from {user.username or user.id}: {user_text[:100]}")

        history = self._history.load(user_id)

        try:
            response_text, updated_history = await self._stream_think(
                update, user_text, history
            )
            self._history.save(user_id, updated_history)

            # Send any files queued by the send_file tool
            for path, caption in self.brain.take_files():
                await self._send_file(update, path, caption)

        except Exception as e:
            logger.exception("Error in brain.think()")
            await update.message.reply_text(
                f"⚠️ Something went wrong: {e}\n\nPlease try again."
            )

    async def _stream_think(
        self,
        update: Update,
        user_message: str | list,
        history: list,
    ) -> tuple[str, list]:
        """
        Call brain.think() with streaming and edit the reply message in real-time.

        Sends a placeholder "…" message immediately, then edits it as tokens
        arrive (throttled to _STREAM_EDIT_INTERVAL seconds between edits).
        The final edit applies proper HTML formatting.

        Returns (response_text, updated_history).
        """
        placeholder = await update.message.reply_text("…")

        last_edit_at: float = 0.0

        async def on_status(status: str) -> None:
            nonlocal last_edit_at
            now = time.monotonic()
            if now - last_edit_at >= _STREAM_EDIT_INTERVAL:
                try:
                    await placeholder.edit_text(status)
                    last_edit_at = now
                except Exception:
                    pass  # Ignore edit failures (unchanged text, network hiccup, etc.)

        response_text, updated_history = await self.brain.think(
            user_message=user_message,
            conversation_history=history,
            on_status=on_status,
        )

        # Final edit: apply HTML formatting and attach keyboard
        final_html = md_to_html(response_text)
        try:
            await placeholder.edit_text(
                final_html, parse_mode="HTML", reply_markup=_MAIN_KEYBOARD
            )
        except Exception:
            # If edit fails (e.g. message too long), fall back to a fresh reply
            await placeholder.delete()
            await self._reply(update, response_text, reply_markup=_MAIN_KEYBOARD)

        return response_text, updated_history

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    async def _send_file(self, update: Update, path, caption: str) -> None:
        """Send a file to the user — as a photo if it's an image, document otherwise."""

        path = Path(path)
        if not path.exists():
            await update.message.reply_text(f"⚠️ File not found: {path.name}")
            return
        try:
            with open(path, "rb") as f:
                if path.suffix.lower() in self._IMAGE_SUFFIXES:
                    await update.message.reply_photo(f, caption=caption or None)
                else:
                    await update.message.reply_document(
                        f, caption=caption or None, filename=path.name
                    )
            logger.info(f"Sent file to user: {path.name}")
        except Exception as e:
            logger.error(f"Failed to send file {path.name}: {e}")
            await update.message.reply_text(f"⚠️ Could not send file '{path.name}': {e}")

    async def push_message(self, text: str, files: list | None = None) -> None:
        """
        Proactively send a message to all allowed users (called by the heartbeat).

        Args:
            text:  Message text to send.
            files: Optional list of (Path, caption) tuples to send as attachments.
        """

        if not self._allowed_users:
            logger.warning(
                "push_message: TELEGRAM_ALLOWED_USERS is empty — nowhere to push"
            )
            return

        for user_id in self._allowed_users:
            try:
                html_text = md_to_html(text)
                for i in range(0, max(1, len(html_text)), 4096):
                    await self.app.bot.send_message(
                        chat_id=user_id, text=html_text[i : i + 4096], parse_mode="HTML"
                    )
                # Send any attached files
                for path, caption in files or []:
                    path = Path(path)
                    if not path.exists():
                        continue
                    with open(path, "rb") as f:
                        if path.suffix.lower() in self._IMAGE_SUFFIXES:
                            await self.app.bot.send_photo(
                                chat_id=user_id, photo=f, caption=caption or None
                            )
                        else:
                            await self.app.bot.send_document(
                                chat_id=user_id,
                                document=f,
                                caption=caption or None,
                                filename=path.name,
                            )
            except Exception as e:
                logger.error(f"push_message: failed to notify user {user_id}: {e}")

    async def start(self):
        """Start the bot (polling mode)."""
        logger.info("Starting Telegram bot (polling)...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot is running. Send a message to start!")

    async def stop(self):
        """Gracefully stop the bot."""
        logger.info("Stopping Telegram bot...")
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
