"""run_job(): the only place tidal_download and vocal_remove meet.

    queued -> downloading -> separating -> done

Both libraries raise typed errors, so failure handling here is a lookup table
rather than logic. Every exception is recorded on the job and swallowed: the
worker loop must survive one bad job.
"""
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Mapping

import tidal_download as td
import vocal_remove as vr

from .jobs import Job, Stage
from . import jobs as jobs_repo

logger = logging.getLogger("app.pipeline")

#: How often the separating job's progress is read and written back.
_PROGRESS_POLL_SECONDS = 0.5


def run_job(conn, job_id: str, client: td.TidalClient,
            models: Mapping[str, vr.LoadedModel], settings) -> Job:
    """Take one queued job through to done or failed.

    Returns the finished Job, and raises only if recording the failure itself
    failed - every error from the work is written to the job instead.
    """
    job = jobs_repo.get(conn, job_id)
    if job is None:
        raise KeyError(f"no such job: {job_id}")

    staging = Path(settings.staging_dir) / job.id
    out_dir = Path(settings.out_dir) / job.id

    try:
        download = _download(conn, job, client, staging)
        stems = _separate(conn, job, download, models, out_dir)
        # Inside the try on purpose: completing is where the unique cache
        # index can fire, and that must be explained rather than escaping as
        # an unhandled error.
        return jobs_repo.advance(
            conn, job.id, Stage.DONE,
            vocals_path=str(stems.vocals),
            instrumental_path=str(stems.instrumental),
        )
    except sqlite3.IntegrityError:
        # Another job for this (track, model, format) completed while this one
        # was running, so the unique cache index rejects this one. The work is
        # wasted; callers prevent it by deduping submissions against
        # jobs.find_active() before creating a job at all.
        logger.warning("job %s duplicated an existing result; discarding", job.id)
        _cleanup(out_dir)
        return jobs_repo.advance(
            conn, job.id, Stage.FAILED,
            error=("a completed result for this track, model and format "
                   "already exists - this duplicate job was discarded"),
        )
    except Exception as exc:
        message = _explain(exc)
        logger.warning("job %s failed: %s", job.id, message)
        _cleanup(out_dir)
        return jobs_repo.advance(conn, job.id, Stage.FAILED, error=message)
    finally:
        # The source is never kept: with results cached, the only thing it
        # would save is a re-download when separating the same track with a
        # different model, which is a different cache key anyway.
        _cleanup(staging)


def _download(conn, job: Job, client: td.TidalClient,
              staging: Path) -> td.DownloadResult:
    jobs_repo.advance(conn, job.id, Stage.DOWNLOADING)

    last = [-1]

    def on_progress(p: td.Progress) -> None:
        # Throttle writes: the callback fires per 64 KiB chunk, which would
        # otherwise be thousands of UPDATEs per track.
        if not p.total:
            return
        pct = int(p.fraction * 100)
        if pct != last[0]:
            last[0] = pct
            jobs_repo.set_progress(conn, job.id, p.fraction)

    result = client.download(job.url, staging, progress=on_progress)

    jobs_repo.set_track_info(
        conn, job.id,
        track_id=result.track.id, title=result.track.title,
        artist=result.track.artist, album=result.track.album,
        duration=result.track.duration,
    )
    logger.info("job %s downloaded %s (%s, %.1f MiB)", job.id,
                result.track.display, result.quality,
                result.size_bytes / 2 ** 20)
    return result


def _separate(conn, job: Job, download: td.DownloadResult,
              models: Mapping[str, vr.LoadedModel],
              out_dir: Path) -> vr.SeparateResult:
    model = models.get(job.model)
    if model is None:
        raise vr.ModelLoadError(
            f"model {job.model!r} is not loaded; available: {sorted(models)}")

    jobs_repo.advance(conn, job.id, Stage.SEPARATING)

    sep_config = vr.SeparationConfig(output_dir=out_dir,
                                     output_format=job.output_format)
    logger.info("job %s separating with %s", job.id, job.model)

    # separate() returns a handle immediately; the work runs on its own daemon
    # thread inside vocal_remove.
    handle = vr.separate(download.path, model, sep_config)
    last_pct = -1

    while not handle.wait(_PROGRESS_POLL_SECONDS):
        fraction = handle.get_progress()
        # Throttle writes to whole percent, as the download does: the poll is
        # far finer-grained than anything the page can show.
        pct = int(fraction * 100)
        if pct != last_pct:
            last_pct = pct
            jobs_repo.set_progress(conn, job.id, fraction)

    return handle.result()


def _explain(exc: Exception) -> str:
    """Turn a library exception into something worth showing a user."""
    if isinstance(exc, td.AuthError):
        return ("not logged in to Tidal - run: "
                "venv\\Scripts\\python.exe -m smoke_test.tidal_cli login")
    if isinstance(exc, td.UnsupportedUrl):
        return f"unsupported URL: {exc}"
    if isinstance(exc, td.NotFound):
        return f"track not found: {exc}"
    if isinstance(exc, td.DownloadError):
        return f"download failed: {exc}"
    if isinstance(exc, vr.OutOfMemory):
        return f"GPU out of memory: {exc}"
    if isinstance(exc, vr.ModelLoadError):
        return f"model unavailable: {exc}"
    if isinstance(exc, (vr.SeparationError, vr.AudioNotFound)):
        return f"separation failed: {exc}"
    logger.exception("unexpected pipeline failure")
    return f"unexpected error: {type(exc).__name__}: {exc}"


def _cleanup(staging: Path) -> None:
    try:
        if staging.is_dir():
            shutil.rmtree(staging, ignore_errors=True)
    except OSError:
        logger.debug("could not remove staging dir %s", staging, exc_info=True)
