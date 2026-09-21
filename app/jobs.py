"""Job records and their state machine.

Imports neither FastAPI nor torch on purpose: this is the part worth testing
exhaustively, and it should not cost a 2 GiB model load to do so.

Lifecycle:

    queued -> downloading -> separating -> done
       \\          \\             \\
        `-----------`-------------`--> failed

`done` and `failed` are terminal.
"""
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Stage(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    SEPARATING = "separating"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (Stage.DONE, Stage.FAILED)


#: Which transitions are legal. Enforced so a bug cannot, say, move a failed
#: job back into separating and have the worker pick it up forever.
_ALLOWED = {
    Stage.QUEUED: {Stage.DOWNLOADING, Stage.FAILED},
    Stage.DOWNLOADING: {Stage.SEPARATING, Stage.FAILED},
    Stage.SEPARATING: {Stage.DONE, Stage.FAILED},
    Stage.DONE: set(),
    Stage.FAILED: set(),
}


class IllegalTransition(Exception):
    """Attempted a stage change the lifecycle does not allow."""


@dataclass
class Job:
    id: str
    url: str
    model: str
    output_format: str
    stage: Stage
    progress: float = 0.0
    error: Optional[str] = None

    track_id: Optional[int] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[int] = None

    vocals_path: Optional[str] = None
    instrumental_path: Optional[str] = None

    created_at: int = 0
    updated_at: int = 0
    started_at: Optional[int] = None
    finished_at: Optional[int] = None

    @property
    def display(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.url

    @property
    def stems_exist(self) -> bool:
        """Both stems still on disk.

        A `done` row can outlive its files - data/out cleared by hand, say -
        so a cache hit must verify rather than trust the database.
        """
        return bool(
            self.vocals_path and self.instrumental_path
            and Path(self.vocals_path).is_file()
            and Path(self.instrumental_path).is_file()
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        d["display"] = self.display
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            id=row["id"], url=row["url"], model=row["model"],
            output_format=row["output_format"], stage=Stage(row["stage"]),
            progress=row["progress"], error=row["error"],
            track_id=row["track_id"], title=row["title"], artist=row["artist"],
            album=row["album"], duration=row["duration"],
            vocals_path=row["vocals_path"],
            instrumental_path=row["instrumental_path"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            started_at=row["started_at"], finished_at=row["finished_at"],
        )


def _now() -> int:
    return int(time.time())


# ------------------------------------------------------------------- writes


def create(conn: sqlite3.Connection, *, url: str, model: str,
           output_format: str, track_id: Optional[int] = None,
           title: Optional[str] = None, artist: Optional[str] = None,
           album: Optional[str] = None,
           duration: Optional[int] = None) -> Job:
    now = _now()
    job = Job(
        id=str(uuid.uuid4()), url=url, model=model,
        output_format=output_format, stage=Stage.QUEUED,
        track_id=track_id, title=title, artist=artist, album=album,
        duration=duration, created_at=now, updated_at=now,
    )
    with conn:
        conn.execute(
            """INSERT INTO jobs (id, url, track_id, title, artist, album,
                                 duration, model, output_format, stage,
                                 progress, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job.id, job.url, job.track_id, job.title, job.artist, job.album,
             job.duration, job.model, job.output_format, job.stage.value,
             job.progress, job.created_at, job.updated_at),
        )
    return job


def advance(conn: sqlite3.Connection, job_id: str, to: Stage, *,
            error: Optional[str] = None,
            vocals_path: Optional[str] = None,
            instrumental_path: Optional[str] = None) -> Job:
    """Move a job to `to`, rejecting transitions the lifecycle forbids."""
    current = get(conn, job_id)
    if current is None:
        raise KeyError(f"no such job: {job_id}")
    if to not in _ALLOWED[current.stage]:
        raise IllegalTransition(
            f"{job_id}: {current.stage.value} -> {to.value} is not allowed"
        )

    now = _now()
    started = current.started_at or (now if to is Stage.DOWNLOADING else None)
    finished = now if to.is_terminal else None
    # Reaching a terminal stage means the bar is either full or irrelevant.
    progress = 1.0 if to is Stage.DONE else (0.0 if to is Stage.SEPARATING
                                             else current.progress)
    with conn:
        conn.execute(
            """UPDATE jobs SET stage=?, error=?, progress=?, updated_at=?,
                               started_at=?, finished_at=?,
                               vocals_path=COALESCE(?, vocals_path),
                               instrumental_path=COALESCE(?, instrumental_path)
               WHERE id=?""",
            (to.value, error, progress, now, started, finished,
             vocals_path, instrumental_path, job_id),
        )
    return get(conn, job_id)


def set_progress(conn: sqlite3.Connection, job_id: str, fraction: float) -> None:
    """Update download progress. Cheap and frequent - no read-back."""
    with conn:
        conn.execute(
            "UPDATE jobs SET progress=?, updated_at=? WHERE id=?",
            (max(0.0, min(1.0, fraction)), _now(), job_id),
        )


def set_track_info(conn: sqlite3.Connection, job_id: str, *,
                   track_id: Optional[int], title: Optional[str],
                   artist: Optional[str], album: Optional[str],
                   duration: Optional[int]) -> None:
    with conn:
        conn.execute(
            """UPDATE jobs SET track_id=?, title=?, artist=?, album=?,
                               duration=?, updated_at=? WHERE id=?""",
            (track_id, title, artist, album, duration, _now(), job_id),
        )


# -------------------------------------------------------------------- reads


def get(conn: sqlite3.Connection, job_id: str) -> Optional[Job]:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def recent(conn: sqlite3.Connection, limit: int = 25) -> List[Job]:
    rows = conn.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [Job.from_row(r) for r in rows]


def next_queued(conn: sqlite3.Connection) -> Optional[Job]:
    row = conn.execute(
        "SELECT * FROM jobs WHERE stage=? ORDER BY created_at ASC LIMIT 1",
        (Stage.QUEUED.value,),
    ).fetchone()
    return Job.from_row(row) if row else None


#: Stages where a job is claimed but not finished.
ACTIVE_STAGES = (Stage.QUEUED, Stage.DOWNLOADING, Stage.SEPARATING)


def find_active(conn: sqlite3.Connection, *, track_id: int, model: str,
                output_format: str) -> Optional[Job]:
    """An in-flight job for this key, if any.

    Submissions dedupe against this. Without it, two jobs for the same track
    can both queue before either completes, and the second violates the unique
    cache index when it finishes.
    """
    row = conn.execute(
        """SELECT * FROM jobs
            WHERE track_id=? AND model=? AND output_format=?
              AND stage IN (?,?,?)
            ORDER BY created_at ASC LIMIT 1""",
        (track_id, model, output_format, *[s.value for s in ACTIVE_STAGES]),
    ).fetchone()
    return Job.from_row(row) if row else None


def find_cached(conn: sqlite3.Connection, *, track_id: int, model: str,
                output_format: str, ttl_seconds: Optional[int]) -> Optional[Job]:
    """The completed job for this key, if one is still usable.

    ttl_seconds: None = permanent, 0 = caching disabled, N = expire after N.

    An entry that has expired, or whose stems have vanished from disk, is
    DELETED here and reported as a miss. That keeps the unique cache index
    from blocking the re-run, and stops a stale row masking missing files.
    """
    row = conn.execute(
        """SELECT * FROM jobs
            WHERE track_id=? AND model=? AND output_format=? AND stage=?""",
        (track_id, model, output_format, Stage.DONE.value),
    ).fetchone()
    if row is None:
        return None
    job = Job.from_row(row)

    if ttl_seconds == 0:
        return None  # caching disabled: keep the row, just do not use it
    expired = (ttl_seconds is not None
               and job.created_at < _now() - ttl_seconds)
    if expired or not job.stems_exist:
        _evict(conn, job)
        return None
    return job


def _evict(conn: sqlite3.Connection, job: Job) -> None:
    """Remove a cache entry and the stems it owns.

    Deleting the row is what frees the unique index for a fresh run. The stem
    files go with it - leaving them would leak disk that nothing references,
    since the row was the only record of where they were.
    """
    for path_str in (job.vocals_path, job.instrumental_path):
        if not path_str:
            continue
        path = Path(path_str)
        try:
            path.unlink(missing_ok=True)
            # Jobs own their output directory; drop it once it is empty.
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except OSError:
            pass  # a leaked file is not worth failing the request over
    with conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job.id,))


def reset_orphans(conn: sqlite3.Connection) -> int:
    """Fail jobs left mid-flight by a crash or restart.

    The worker is the only thing that advances a job past `queued`, so
    anything found in an active stage at startup belongs to a dead process.
    """
    with conn:
        cur = conn.execute(
            """UPDATE jobs SET stage=?, error=?, updated_at=?, finished_at=?
                WHERE stage IN (?, ?)""",
            (Stage.FAILED.value, "interrupted by a restart", _now(), _now(),
             Stage.DOWNLOADING.value, Stage.SEPARATING.value),
        )
    return cur.rowcount
