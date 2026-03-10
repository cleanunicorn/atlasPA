"""
channels/telegram/bot.py

Telegram channel adapter.
Receives messages from Telegram → forwards to Brain → sends response back.

Security:
    Only responds to user IDs listed in TELEGRAM_ALLOWED_USERS.
    Set this in your .env to prevent others from using your agent.
"""

import logging
import os
from pathlib import Path
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from memory.history import ConversationHistory

# Where incoming files are staged before the brain decides what to do with them
_UPLOAD_DIR = Path.home() / "agent-files" / "uploads"

logger = logging.getLogger(__name__)

# Persistent reply keyboard shown to all authorised users
_MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton("/clear"), KeyboardButton("/status")]],
    resize_keyboard=True,
    is_persistent=True,
)


class TelegramBot:
    def __init__(self, brain, on_message_callback=None):
        """
        Args:
            brain:                  The Brain instance.
            on_message_callback:    Optional async callback for routing (used by Gateway).
        """
        self.brain = brain
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")

        # Persistent conversation history (survives restarts)
        self._history = ConversationHistory()

        # Allowed user IDs (security gate)
        allowed_raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
        self._allowed_users: set[int] = set()
        if allowed_raw.strip():
            for uid in allowed_raw.split(","):
                try:
                    self._allowed_users.add(int(uid.strip()))
                except ValueError:
                    logger.warning(f"Invalid user ID in TELEGRAM_ALLOWED_USERS: {uid}")

        if not self._allowed_users:
            logger.warning(
                "⚠️  TELEGRAM_ALLOWED_USERS is empty — bot will respond to ANYONE. "
                "Set this to your Telegram user ID for security."
            )

        self.app = Application.builder().token(self.token).build()
        self._register_handlers()

    def _is_allowed(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _register_handlers(self):
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("clear", self._cmd_clear))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        self.app.add_handler(
            MessageHandler(filters.Document.ALL, self._handle_document)
        )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        agent_name = os.getenv("AGENT_NAME", "Atlas")
        history_len = len(self._history.load(str(user.id)))
        resume_note = (
            f" (resuming — {history_len} messages in history)"
            if history_len
            else ""
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
        await update.message.reply_text("🧹 Conversation cleared.", reply_markup=_MAIN_KEYBOARD)

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        history = self._history.load(str(user_id))
        provider_name = self.brain.provider.model_name
        skills = self.brain.skills.all_skill_names()
        context_entries = len(self.brain.memory.parse_context_entries())
        await update.message.reply_text(
            f"🤖 **Agent Status**\n"
            f"Model: `{provider_name}`\n"
            f"Conversation: {len(history)} messages\n"
            f"Long-term memories: {context_entries}\n"
            f"Skills: {', '.join(skills) or 'none'}",
            reply_markup=_MAIN_KEYBOARD,
        )

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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

        logger.info(f"File received from {user.username or user.id}: {filename} → {save_path}")

        # Build a context message for the brain
        user_message = (
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

            if len(response_text) > 4000:
                chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for i, chunk in enumerate(chunks):
                    markup = _MAIN_KEYBOARD if i == len(chunks) - 1 else None
                    await update.message.reply_text(chunk, reply_markup=markup)
            else:
                await update.message.reply_text(response_text, reply_markup=_MAIN_KEYBOARD)

            for path, caption_f in self.brain.take_files():
                await self._send_file(update, path, caption_f)

        except Exception as e:
            logger.exception("Error processing uploaded file")
            await update.message.reply_text(f"⚠️ Something went wrong: {e}")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not self._is_allowed(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            return

        user_text = update.message.text
        user_id = str(user.id)

        logger.info(f"Message from {user.username or user.id}: {user_text[:100]}")

        await update.message.chat.send_action("typing")

        history = self._history.load(user_id)

        try:
            response_text, updated_history = await self.brain.think(
                user_message=user_text,
                conversation_history=history,
            )
            self._history.save(user_id, updated_history)

            if len(response_text) > 4000:
                chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for i, chunk in enumerate(chunks):
                    markup = _MAIN_KEYBOARD if i == len(chunks) - 1 else None
                    await update.message.reply_text(chunk, reply_markup=markup)
            else:
                await update.message.reply_text(response_text, reply_markup=_MAIN_KEYBOARD)

            # Send any files queued by the send_file tool
            for path, caption in self.brain.take_files():
                await self._send_file(update, path, caption)

        except Exception as e:
            logger.exception("Error in brain.think()")
            await update.message.reply_text(
                f"⚠️ Something went wrong: {e}\n\nPlease try again."
            )

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

    async def _send_file(self, update: Update, path, caption: str) -> None:
        """Send a file to the user — as a photo if it's an image, document otherwise."""
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            await update.message.reply_text(f"⚠️ File not found: {path.name}")
            return
        try:
            with open(path, "rb") as f:
                if path.suffix.lower() in self._IMAGE_SUFFIXES:
                    await update.message.reply_photo(f, caption=caption or None)
                else:
                    await update.message.reply_document(f, caption=caption or None, filename=path.name)
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
        from pathlib import Path

        if not self._allowed_users:
            logger.warning("push_message: TELEGRAM_ALLOWED_USERS is empty — nowhere to push")
            return

        for user_id in self._allowed_users:
            try:
                # Split long messages
                for i in range(0, max(1, len(text)), 4000):
                    await self.app.bot.send_message(chat_id=user_id, text=text[i:i+4000])
                # Send any attached files
                for path, caption in (files or []):
                    path = Path(path)
                    if not path.exists():
                        continue
                    with open(path, "rb") as f:
                        if path.suffix.lower() in self._IMAGE_SUFFIXES:
                            await self.app.bot.send_photo(chat_id=user_id, photo=f, caption=caption or None)
                        else:
                            await self.app.bot.send_document(chat_id=user_id, document=f, caption=caption or None, filename=path.name)
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
