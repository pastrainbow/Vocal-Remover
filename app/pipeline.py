"""run_job(): the only place tidal_download and vocal_remove meet.

    queued -> downloading -> separating -> done

Both libraries raise typed errors, so failure handling here is a lookup table
rather than logic. Every exception is recorded on the job and swallowed: the
worker loop must survive one bad job.
"""
import logging
import shutil
import sqlite3
import threading
from pathlib import Path
from typing import Mapping, Optional

import tidal_download as td
import vocal_remove as vr

from .jobs import Job, Stage
from . import jobs as jobs_repo

logger = logging.getLogger("app.pipeline")

#: Measured on an RTX 4060 with a 229.5s track: how many seconds of audio each
#: model separates per second of wall clock. Used only to size the timeout.
_REALTIME_MULTIPLIER = {
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt": 1.8,   # 126.5s
    "UVR-MDX-NET-Voc_FT.onnx": 11.7,                    # 19.6s
}
#: Unknown models are assumed slower than anything measured, so an unfamiliar
#: model gets a generous timeout rather than a spurious failure.
_UNKNOWN_MULTIPLIER = 1.0
#: When track duration is unknown there is nothing to scale, so allow a lot.
_NO_DURATION_TIMEOUT = 30 * 60


class WorkerCompromised(RuntimeError):
    """Separation exceeded its timeout and could not be cancelled.

    separate() blocks inside CUDA/C, so a Python thread cannot interrupt it.
    The job is failed, but the runaway thread still holds the GPU, so the
    worker process must restart to get back to a known state.
    """


def separation_timeout(duration_seconds: Optional[int], model: str,
                       factor: float, floor_seconds: int) -> float:
    """How long separation is allowed before the job is abandoned.

    A stand-in for real progress: separate() reports nothing until it returns
    (see the TODO on vocal_remove.separate), so a stalled job is otherwise
    indistinguishable from a slow one. Delete this once chunk counts exist.
    """
    if not duration_seconds:
        return float(_NO_DURATION_TIMEOUT)
    multiplier = _REALTIME_MULTIPLIER.get(model, _UNKNOWN_MULTIPLIER)
    return max(float(floor_seconds), (duration_seconds / multiplier) * factor)


def run_job(conn, job_id: str, client: td.TidalClient,
            models: Mapping[str, vr.LoadedModel], settings) -> Job:
    """Take one queued job through to done or failed.

    Returns the finished Job. Raises only WorkerCompromised, which the caller
    must treat as fatal to the process.
    """
    job = jobs_repo.get(conn, job_id)
    if job is None:
        raise KeyError(f"no such job: {job_id}")

    staging = Path(settings.staging_dir) / job.id
    out_dir = Path(settings.out_dir) / job.id

    try:
        download = _download(conn, job, client, staging)
        stems = _separate(conn, job, download, models, out_dir, settings)
        # Inside the try on purpose: completing is where the unique cache
        # index can fire, and that must be explained rather than escaping as
        # an unhandled error.
        return jobs_repo.advance(
            conn, job.id, Stage.DONE,
            vocals_path=str(stems.vocals),
            instrumental_path=str(stems.instrumental),
        )
    except WorkerCompromised:
        raise
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
              models: Mapping[str, vr.LoadedModel], out_dir: Path,
              settings) -> vr.SeparateResult:
    model = models.get(job.model)
    if model is None:
        raise vr.ModelLoadError(
            f"model {job.model!r} is not loaded; available: {sorted(models)}")

    jobs_repo.advance(conn, job.id, Stage.SEPARATING)

    _, sep_cls = vr.config_classes_for(job.model)
    sep_config = sep_cls(output_dir=out_dir, output_format=job.output_format)

    timeout = separation_timeout(
        download.track.duration, job.model,
        settings.separation_timeout_factor,
        settings.separation_timeout_floor_seconds,
    )
    logger.info("job %s separating with %s (timeout %.0fs)",
                job.id, job.model, timeout)

    # A bare daemon thread, NOT ThreadPoolExecutor: the executor's context
    # manager (and its atexit hook) join their workers on shutdown, which
    # blocks until a runaway separation finishes - defeating the timeout in
    # precisely the hung case it exists to catch. A daemon thread can be
    # abandoned, and dies with the process.
    box: dict = {}

    def _work():
        try:
            box["result"] = vr.separate(download.path, model, sep_config)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
            box["error"] = exc

    thread = threading.Thread(target=_work, name=f"sep-{job.id[:8]}",
                              daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        jobs_repo.advance(
            conn, job.id, Stage.FAILED,
            error=f"separation exceeded {timeout:.0f}s and was abandoned",
        )
        raise WorkerCompromised(
            f"job {job.id}: separation timed out after {timeout:.0f}s; the "
            f"thread cannot be cancelled and still holds the GPU, so the "
            f"worker must restart"
        )

    if "error" in box:
        raise box["error"]
    return box["result"]


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
