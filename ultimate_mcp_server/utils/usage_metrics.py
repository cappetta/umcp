"""SQLite-backed logging for model usage metrics."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from ultimate_mcp_server.config import get_config
from ultimate_mcp_server.utils import get_logger

_logger = get_logger(__name__)
_DB_LOCK = Lock()
_DB_INITIALIZED = False
_DB_PATH: Optional[Path] = None


def _get_db_path() -> Path:
    """Return the filesystem path for the usage metrics database."""
    global _DB_PATH
    if _DB_PATH is None:
        config = get_config()
        storage_dir = Path(config.storage_directory)
        storage_dir.mkdir(parents=True, exist_ok=True)
        _DB_PATH = storage_dir / "usage_metrics.db"
    return _DB_PATH


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the metrics table exists."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            cost REAL,
            processing_time REAL,
            metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_model_usage_timestamp
            ON model_usage (timestamp)
        """
    )


def log_model_usage(
    *,
    provider: Optional[str],
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cost: Optional[float],
    processing_time: Optional[float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a single usage record to the metrics database.

    Errors are logged but never propagated to avoid disrupting normal flows.
    """
    global _DB_INITIALIZED

    try:
        metadata_payload: Optional[str]
        if metadata:
            try:
                metadata_payload = json.dumps(metadata, default=str)
            except TypeError:
                metadata_payload = json.dumps({"repr": repr(metadata)})
        else:
            metadata_payload = None

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        db_path = _get_db_path()

        with _DB_LOCK:
            conn = sqlite3.connect(str(db_path), timeout=5)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                if not _DB_INITIALIZED:
                    _ensure_schema(conn)
                    _DB_INITIALIZED = True
                conn.execute(
                    """
                    INSERT INTO model_usage (
                        timestamp,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost,
                        processing_time,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        provider,
                        model,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost,
                        processing_time,
                        metadata_payload,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except sqlite3.Error as exc:
        _logger.error("Failed to log model usage to SQLite: %s", exc, exc_info=True, emoji_key="database")
    except Exception:
        _logger.debug("Unexpected error while logging model usage", exc_info=True)
