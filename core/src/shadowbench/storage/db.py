"""SQLite access layer.  [Phase 3 — P3.5]

Thin wrapper around the ``schema.sql`` tables. Every completed run logs predicted vs. actual t/s here; the
calibration loop reads from it (locally) and, if opted in, syncs anonymized rows upstream.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the local database with the schema applied."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _apply_schema(conn)
    return conn


def _apply_schema(conn: sqlite3.Connection) -> None:
    schema = (
        resources.files("shadowbench.storage").joinpath("schema.sql").read_text(encoding="utf-8")
    )
    conn.executescript(schema)
    conn.commit()
