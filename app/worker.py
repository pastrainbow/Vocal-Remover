"""The background worker: holds models resident and drains the job queue.

One thread, one job at a time. Separation is GPU-bound and torch releases the
GIL during inference, so a thread rather than a process keeps the loaded models
reachable without any IPC - which is the whole reason init_models() exists.

The tradeoff is that separation is not isolated from the web process: anything
that takes the separation thread down hard, a CUDA fault in particular, takes
the server with it.
"""
import logging
import threading
from typing import Dict, List, Optional

import tidal_download as td
import vocal_remove as vr

from . import db, jobs as jobs_repo, pipeline
from .config import Settings

logger = logging.getLogger("app.worker")

#: How long the loop sleeps when the queue is empty. Submissions call notify()
#: to wake it immediately, so this is only a backstop.
_IDLE_POLL_SECONDS = 2.0


class ModelsUnavailable(RuntimeError):
    """One or more configured models could not be loaded.

    Raised at startup rather than degrading: preloading everything up front is
    the point of the design, so a model that will not load is a configuration
    error to fix, not a condition to work around.
    """


class Worker:
    """Owns the models and the queue-draining thread."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: Dict[str, vr.LoadedModel] = {}
        self.load_report: List[vr.LoadedModel] = []

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._client: Optional[td.TidalClient] = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Load models, recover orphaned jobs, then start the loop.

        Models load synchronously so a failure surfaces as a refusal to start
        rather than as jobs failing one by one later.
        """
        self.settings.ensure_dirs()
        self._client = td.TidalClient(config_dir=self.settings.state_dir / "tidal")

        configs = []
        for name in self.settings.preload_models:
            model_cls, _ = vr.config_classes_for(name)
            configs.append(model_cls(
                name=name,
                model_dir=self.settings.models_dir,
                segment_size=self.settings.segment_size,
            ))

        self.load_report = vr.init_models(configs)
        self.models = {m.config.name: m for m in self.load_report if m.ok}

        failed = [m for m in self.load_report
                  if m.status is vr.LoadStatus.FAILED]
        if failed:
            detail = "; ".join(f"{m.config.name}: {m.error}" for m in failed)
            raise ModelsUnavailable(
                f"{len(failed)} of {len(configs)} models failed to load - "
                f"{detail}"
            )
        duplicates = [m.config.name for m in self.load_report
                      if m.status is vr.LoadStatus.DUPLICATE]
        if duplicates:
            logger.warning("ignoring duplicate entries in preload_models: %s",
                           ", ".join(duplicates))

        cpu_only = [n for n, m in self.models.items() if not m.on_gpu]
        if cpu_only:
            logger.warning("models running on CPU, expect very slow jobs: %s",
                           ", ".join(cpu_only))

        # NB: sqlite3's context manager commits, it does not close - so this
        # is deliberately explicit rather than a `with` block.
        conn = db.init_db(self.settings.db_path)
        try:
            orphaned = jobs_repo.reset_orphans(conn)
        finally:
            conn.close()
        if orphaned:
            logger.warning("failed %d job(s) left mid-flight by a restart",
                           orphaned)

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="separator",
                                        daemon=True)
        self._thread.start()
        logger.info("worker started with %d model(s): %s",
                    len(self.models), ", ".join(self.models))

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the loop to finish. A job already running is allowed to end."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("worker thread did not stop within %.0fs; it is "
                               "a daemon and will die with the process", timeout)
        self._thread = None

    def notify(self) -> None:
        """Wake the loop immediately - call after queueing a job."""
        self._wake.set()

    # --------------------------------------------------------------- status

    @property
    def client(self) -> Optional[td.TidalClient]:
        """The Tidal session, or None before start(). Shared with the API so
        submissions can validate a URL without building a second session."""
        return self._client

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        return {
            "running": self.running,
            "models": [
                {
                    "name": m.config.name,
                    "status": m.status.value,
                    "device": m.device if m.ok else None,
                    "load_seconds": round(m.load_seconds, 1) if m.ok else None,
                    "error": m.error,
                }
                for m in self.load_report
            ],
        }

    # ----------------------------------------------------------------- loop

    def _run(self) -> None:
        # Own connection: sqlite handles are cheap and this keeps the worker's
        # writes off whatever the request handlers are doing.
        conn = db.connect(self.settings.db_path)
        try:
            while not self._stop.is_set():
                job = jobs_repo.next_queued(conn)
                if job is None:
                    self._wake.wait(timeout=_IDLE_POLL_SECONDS)
                    self._wake.clear()
                    continue
                self._run_one(conn, job.id)
        finally:
            conn.close()
            logger.info("worker loop exited")

    def _run_one(self, conn, job_id: str) -> None:
        try:
            done = pipeline.run_job(conn, job_id, self._client, self.models,
                                    self.settings)
            logger.info("job %s -> %s", job_id[:8], done.stage.value)
        except Exception:
            # run_job records its own failures; reaching here means the failure
            # handling itself broke. Keep the loop alive but do not retry the
            # job, or a poison row would spin forever.
            logger.exception("unhandled error running job %s", job_id)
            try:
                jobs_repo.advance(conn, job_id, jobs_repo.Stage.FAILED,
                                  error="internal worker error")
            except Exception:
                logger.exception("could not mark job %s failed", job_id)
