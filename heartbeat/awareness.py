"""
heartbeat/awareness.py

Autonomous awareness loop — periodically asks the LLM whether it should
proactively do something for the user.

How it works:
    1. Every AWARENESS_INTERVAL_MINUTES the _check() coroutine fires.
    2. A prompt is built that includes recent actions already taken (so the
       LLM won't repeat itself) and the agent's soul / current context.
    3. If the LLM decides there is nothing useful to do it returns the
       sentinel "NO_ACTION" and nothing happens.
    4. Otherwise the response is treated as a real proactive message and
       pushed to the user via notify_callback.
    5. Every check (action or no-action) is appended to
       memory/awareness_log.json, capped at LOG_MAX_ENTRIES entries.

Usage:
    awareness = Awareness(brain=brain, notify_callback=my_async_fn)
    await awareness.start()
    ...
    await awareness.stop()
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# How often the awareness loop ticks (minutes)
AWARENESS_INTERVAL_MINUTES = 30

# Cap on log entries kept in awareness_log.json
LOG_MAX_ENTRIES = 50

# How many recent *triggered* actions to include in the awareness prompt
PROMPT_RECENT_ACTIONS = 5

LOG_FILE = Path(__file__).parent.parent / "memory" / "awareness_log.json"

# Sentinel the LLM should return when no proactive action is needed
NO_ACTION_SENTINEL = "NO_ACTION"

AWARENESS_PROMPT = """\
You are an autonomous awareness module for a personal AI agent.

Your job: decide if there is something proactive and genuinely useful you
should do for the user RIGHT NOW — e.g. a reminder, a helpful insight, a
morning summary, following up on something they mentioned.

You have already taken these recent actions (do NOT repeat them):
{recent_actions}

Rules:
- If there is nothing worthwhile to do, reply with exactly: NO_ACTION
- If there IS something useful, write the proactive message to the user
  (one or two sentences max). Do NOT start with "NO_ACTION".
- Be conservative — do not act unless you are confident it adds value.
"""


def _load_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    try:
        return json.loads(LOG_FILE.read_text())
    except Exception:
        return []


def _save_log(entries: list[dict]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOG_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    tmp.replace(LOG_FILE)


def _append_log(triggered: bool, summary: str) -> None:
    entries = _load_log()
    entries.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "triggered": triggered,
        "summary": summary[:200],
    })
    _save_log(entries[-LOG_MAX_ENTRIES:])


def _recent_action_lines() -> str:
    entries = _load_log()
    triggered = [e for e in entries if e.get("triggered")]
    recent = triggered[-PROMPT_RECENT_ACTIONS:]
    if not recent:
        return "  (none yet)"
    return "\n".join(f"  [{e['ts'][:16]}] {e['summary']}" for e in recent)


class Awareness:
    """
    Periodic autonomous check — uses the LLM to decide whether to act.
    """

    def __init__(self, brain, notify_callback=None):
        """
        Args:
            brain:            Brain instance.
            notify_callback:  async def notify(text, files) -> None
        """
        self.brain = brain
        self.notify_callback = notify_callback
        self._scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        self._scheduler.add_job(
            self._check,
            trigger=IntervalTrigger(minutes=AWARENESS_INTERVAL_MINUTES),
            id="awareness_check",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"Awareness started — checking every {AWARENESS_INTERVAL_MINUTES}m"
        )

    async def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("Awareness stopped")

    async def _check(self) -> None:
        """Run one awareness tick."""
        logger.debug("Awareness tick — asking LLM")
        prompt = AWARENESS_PROMPT.format(recent_actions=_recent_action_lines())

        try:
            response, _ = await self.brain.think(prompt, conversation_history=[])
        except Exception as e:
            logger.exception(f"Awareness LLM call failed: {e}")
            return

        if response.strip().startswith(NO_ACTION_SENTINEL):
            logger.debug("Awareness: NO_ACTION")
            _append_log(triggered=False, summary="no action")
            return

        logger.info(f"Awareness triggered: {response[:80]}")
        _append_log(triggered=True, summary=response[:200])

        files = self.brain.take_files()
        if self.notify_callback:
            try:
                await self.notify_callback(response, files)
            except Exception as e:
                logger.exception(f"Awareness notify failed: {e}")
        else:
            logger.info(f"Awareness response (no callback): {response[:200]}")
