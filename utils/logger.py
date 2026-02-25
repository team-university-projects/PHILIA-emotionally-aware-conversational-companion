"""
logger.py — Structured, levelled logging for PHILIA.

Responsibilities:
  - Provide a named logger factory used by all modules
  - Write to both console (coloured) and a rotating file
  - Respect the log level from Config
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import io

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "philia.log"
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_initialised = False


def _setup_root(level: str = "INFO") -> None:
    """Configure the root logger once."""
    global _initialised
    if _initialised:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler — wrap stdout in UTF-8 so Unicode chars (→, etc.)
    # don't crash on Windows cp1252 terminals.
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(_FORMATTER)
    root.addHandler(ch)

    # Rotating file handler (5 MB × 3 backups)
    fh = RotatingFileHandler(_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(_FORMATTER)
    root.addHandler(fh)

    _initialised = True


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a named logger, initialising the root handler on first call.

    Args:
        name:  Module name, typically __name__.
        level: Logging level string (DEBUG / INFO / WARNING / ERROR).

    Returns:
        logging.Logger instance.
    """
    _setup_root(level)
    return logging.getLogger(name)
