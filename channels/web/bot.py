"""
channels/web/bot.py

Web UI channel adapter — serves a browser-based chat interface.

Architecture:
    - FastAPI app served by uvicorn
    - GET  /          → chat UI (HTML/JS)
    - WS   /ws/{sid}  → WebSocket per browser tab (sid = random session UUID)
    - GET  /files/{name} → download files queued by send_file tool

Each browser tab gets a unique session_id (generated client-side).
Conversation history is persisted per session_id in memory/history/.

Message protocol (JSON over WebSocket):
    Client → server:  {"message": "user text"}
    Server → client:  {"type": "text",  "content": "agent reply"}
                      {"type": "file",  "name": "foo.png", "url": "/files/foo.png"}
                      {"type": "error", "content": "..."}

Required .env:
    WEB_HOST=0.0.0.0   (default)
    WEB_PORT=7860      (default)
"""

import asyncio
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from memory.history import ConversationHistory

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# Files served by /files/<name> come from agent-files dir
_FILES_DIR = Path.home() / "agent-files"


class WebBot:
    """
    Browser-based chat channel powered by FastAPI + WebSockets.

    One WebBot instance is shared across all open browser tabs.
    Each tab has its own session_id and isolated conversation history.
    """

    def __init__(self, brain, host: str = "", port: int = 0):
        self.brain = brain
        self._host = host or os.getenv("WEB_HOST", "0.0.0.0")
        self._port = port or int(os.getenv("WEB_PORT", "7860"))
        self._history = ConversationHistory()
        self._server: uvicorn.Server | None = None

        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Atlas Web UI", docs_url=None, redoc_url=None)

        # Serve agent-generated files (screenshots etc.)
        _FILES_DIR.mkdir(parents=True, exist_ok=True)
        app.mount("/files", StaticFiles(directory=str(_FILES_DIR), check_dir=False), name="files")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            html = (_STATIC_DIR / "index.html").read_text()
            return HTMLResponse(html)

        @app.websocket("/ws/{session_id}")
        async def ws_chat(websocket: WebSocket, session_id: str):
            await websocket.accept()
            logger.info(f"WebSocket connected: session={session_id}")
            history = self._history.load(session_id)

            try:
                while True:
                    data = await websocket.receive_json()
                    user_msg = (data.get("message") or "").strip()
                    if not user_msg:
                        continue

                    # Commands
                    if user_msg.lower() == "/clear":
                        self._history.clear(session_id)
                        history = []
                        await websocket.send_json({"type": "text", "content": "🧹 Conversation cleared."})
                        continue

                    logger.info(f"Web [{session_id[:8]}]: {user_msg[:100]}")

                    try:
                        response, history = await self.brain.think(
                            user_message=user_msg,
                            conversation_history=history,
                        )
                        self._history.save(session_id, history)
                        await websocket.send_json({"type": "text", "content": response})

                        # Send file references
                        for path, caption in self.brain.take_files():
                            path = Path(path)
                            if path.exists():
                                # Copy to agent-files root so it's accessible via /files/
                                dest = _FILES_DIR / path.name
                                if path != dest:
                                    import shutil
                                    shutil.copy2(path, dest)
                                await websocket.send_json({
                                    "type": "file",
                                    "name": path.name,
                                    "url": f"/files/{path.name}",
                                    "caption": caption,
                                })

                    except Exception as e:
                        logger.exception("Error in brain.think()")
                        await websocket.send_json({"type": "error", "content": str(e)})

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: session={session_id}")

        return app

    async def push_message(self, text: str, files: list | None = None) -> None:
        """
        Proactive push from heartbeat.

        The web UI is pull-based (WebSocket per session), so we can't push
        to arbitrary open tabs without a registry. Log the message instead.
        Channels like Telegram or Discord are better for proactive delivery.
        """
        logger.info(f"[WebBot push — no active session to deliver to]: {text[:200]}")

    async def start(self) -> None:
        """Start the uvicorn server in a background task."""
        config = uvicorn.Config(
            self.app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        asyncio.get_event_loop().create_task(self._server.serve())
        logger.info(f"Web UI available at http://{self._host}:{self._port}")

    async def stop(self) -> None:
        """Gracefully shut down the uvicorn server."""
        if self._server:
            self._server.should_exit = True
        logger.info("Web UI stopped")
