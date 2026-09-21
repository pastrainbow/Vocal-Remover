"""SQLite storage.

stdlib sqlite3, no ORM: one table, a handful of queries. WAL mode so SSE
readers never block the worker's writes.

The cache index is UNIQUE, so at most one completed job can exist per
(track_id, model, output_format). Two things make that safe:

  * an expired entry is deleted on lookup (jobs.find_cached), so it cannot
    block the re-run it is meant to permit;
  * a submission is deduped against in-flight work (jobs.find_active), so two
    jobs for the same key never race to completion - without that, the second
    one violates the constraint when it finishes.
"""
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    url             TEXT NOT NULL,

    track_id        INTEGER,
    title           TEXT,
    artist          TEXT,
    album           TEXT,
    duration        INTEGER,

    model           TEXT NOT NULL,
    output_format   TEXT NOT NULL,

    stage           TEXT NOT NULL,
    progress        REAL NOT NULL DEFAULT 0,
    error           TEXT,

    vocals_path        TEXT,
    instrumental_path  TEXT,

    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    started_at      INTEGER,
    finished_at     INTEGER
);

-- At most one completed job per track/model/format. See module docstring for
-- why this is safe alongside a finite TTL.
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_cache
    ON jobs(track_id, model, output_format)
    WHERE stage = 'done' AND track_id IS NOT NULL;

-- Dedupe lookup for in-flight work.
CREATE INDEX IF NOT EXISTS ix_jobs_active
    ON jobs(track_id, model, output_format, stage);

-- Worker queue scan and the recent-jobs listing.
CREATE INDEX IF NOT EXISTS ix_jobs_stage_created
    ON jobs(stage, created_at);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this app depends on.

    check_same_thread=False because the worker thread and request handlers
    share connections through short-lived `with` blocks; every write is a
    single statement inside a transaction, so SQLite's own locking suffices.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Wait rather than failing instantly if the worker holds a write lock.
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the schema if absent and return an open connection."""
    conn = connect(db_path)
    with conn:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    return conn


def schema_version(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute("PRAGMA user_version").fetchone()
    return row[0] if row else None
