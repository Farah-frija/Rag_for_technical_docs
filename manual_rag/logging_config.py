"""
logging_config.py
------------------
One place that configures logging for the entire project.

Replaces the pattern in the original codebase where every module
(`retriever.py`, `client_llm.py`, `summarize_tree.py`, `page_index_approach.py`, ...)
built its own `FileHandler` + formatter from scratch. Now every module just
does:

    from manual_rag.logging_config import get_logger
    logger = get_logger(__name__)

and gets a consistently formatted, correctly routed logger for free:

  * `logs/app.log`        — everything, human-readable (or JSON if configured)
  * `logs/llm_calls.log`  — every LLM prompt + structured output + token usage
                             (only the "manual_rag.llm_calls" logger writes here)
  * console                — INFO+ by default, human-readable

This is what makes the system *observable*: one log directory, one format,
one place to add e.g. a Sentry/OpenTelemetry handler later.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from manual_rag.config import Settings, get_settings

_CONFIGURED = False
LLM_CALLS_LOGGER_NAME = "manual_rag.llm_calls"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Pull through any `extra=` fields the caller attached.
        reserved = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                try:
                    json.dumps(value)
                    payload[key] = value
                except TypeError:
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Idempotent. Call once at process start (CLI does this automatically)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = settings or get_settings()
    log_cfg = settings.logging
    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("manual_rag")
    root.setLevel(log_cfg.level)
    root.propagate = False

    formatter: logging.Formatter
    formatter = _JsonFormatter() if log_cfg.json_logs else logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-32s %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if log_cfg.console_logs:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")
        )
        console.setLevel(logging.INFO)
        root.addHandler(console)

    if log_cfg.log_llm_calls:
        llm_logger = logging.getLogger(LLM_CALLS_LOGGER_NAME)
        llm_logger.setLevel(logging.DEBUG)
        llm_logger.propagate = False
        llm_handler = RotatingFileHandler(
            log_dir / "llm_calls.log", maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        llm_handler.setFormatter(_JsonFormatter())
        llm_logger.addHandler(llm_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `manual_rag` root. Configures
    logging on first use so importing this module is always safe, even
    before the CLI has called `configure_logging()` explicitly."""
    if not _CONFIGURED:
        configure_logging()
    if not name.startswith("manual_rag"):
        name = f"manual_rag.{name}"
    return logging.getLogger(name)


def get_llm_call_logger() -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(LLM_CALLS_LOGGER_NAME)
