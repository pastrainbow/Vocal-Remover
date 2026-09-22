"""The separator, running as its own process.

This is the child half of the worker; Worker in supervise.py is the parent
half that spawns and supervises it. Separation used to run on a thread inside
the web process, which tied the two together: a CUDA fault or a host OOM kill
inside the separator took the server down with it. Here it can die on its own.

Nothing is shared in memory. The two processes meet at four places, and that
is the whole interface:

  * state/app.db - WAL-mode SQLite, already safe for two writers, and the only
    place job state ever lived anyway;
  * state/tidal/session.json - the stored Tidal session. The web process owns
    the login; this one re-reads the file per job, so signing in on the page
    reaches the worker without a message;
  * a one-way pipe carrying the model load report, or the reason there is not
    one;
  * a second one-way pipe, the other way, carrying WAKE and STOP. Reading it
    is also how the loop idles, so a parent that dies without saying anything
    reads as EOF here and this process leaves rather than holding a GPU for a
    server that has gone.

The models load HERE rather than being handed over, which is the point of the
split: the resident weights and the CUDA context belong to the process that
can be lost without taking the server with it.
"""
import logging
import signal
from typing import Dict, List, Mapping, Tuple

import tidal_download as td
import vocal_remove as vr

from .. import db, jobs as jobs_repo, model_settings as model_settings_repo, pipeline
from ..config import LOG_DATEFMT, LOG_FORMAT, PRELOAD_MODELS, Settings
# Importing supervise here is not a cycle: it reaches this module only from
# inside the function it hands to multiprocessing, never at import time.
from .supervise import FATAL, READY, STOP

logger = logging.getLogger("app.worker")

#: How long the loop waits on the command pipe when the queue is empty. A
#: submission sends WAKE to cut this short, so it is only a backstop - and a
#: lost WAKE costs a job this much, not forever.
_IDLE_POLL_SECONDS = 2.0


def run(settings: Settings, report, commands) -> None:
    """Child entry point: load the models, report, then drain the queue.

    Returning from here ends the process. Every exit the parent did not ask
    for is a crash as far as it is concerned, which is exactly the behaviour
    wanted: it restarts and says why.
    """
    # Ctrl-C in the terminal reaches the whole process group. Ignoring it here
    # leaves shutdown to the parent, which stops us deliberately with STOP -
    # otherwise the child dies first and the supervisor, still up, treats a
    # shutdown as a crash and restarts into a closing server.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT,
                        datefmt=LOG_DATEFMT)

    try:
        try:
            models, load_report = _load_models(settings)
        except Exception as exc:
            # Nothing loadable means nothing to do: report the reason and go.
            # The parent turns that into a refusal to start the server, or -
            # if it happens on a restart - into a sentence on /api/health and
            # a backoff, rather than a silent crash loop against a bad file.
            logger.exception("worker process could not start")
            _send(report, FATAL, f"{type(exc).__name__}: {exc}")
            return

        _send(report, READY, load_report)
        logger.info("worker process ready with %d model(s): %s",
                    len(models), ", ".join(models))
        _drain(settings, models, commands)
    finally:
        report.close()
        commands.close()


# ------------------------------------------------------------------- models


class ModelsUnavailable(RuntimeError):
    """One or more configured models could not be loaded.

    Raised and caught inside this process; the parent only ever sees the
    message. A LoadedModel holds a live Separator and never crosses the pipe.
    """


def _load_models(settings: Settings) -> Tuple[Dict[str, vr.LoadedModel],
                                              List[dict]]:
    """Bring up every configured model, or raise saying which would not.

    Preloading everything up front is the design, so a model that will not
    load is a configuration error to fix rather than a condition to work
    around - hence a refusal rather than a degraded worker.
    """
    settings.ensure_dirs()

    configs = []
    for name in PRELOAD_MODELS:
        model_cls, _ = vr.config_classes_for(name)
        params = model_settings_repo.load_params(settings.state_dir, name)
        configs.append(model_cls(
            name=name,
            model_dir=settings.models_dir,
            **params.load_time_kwargs(),
        ))

    loaded = vr.init_models(configs)
    models = {m.config.name: m for m in loaded if m.ok}

    failed = [m for m in loaded if m.status is vr.LoadStatus.FAILED]
    if failed:
        detail = "; ".join(f"{m.config.name}: {m.error}" for m in failed)
        raise ModelsUnavailable(
            f"{len(failed)} of {len(configs)} models failed to load - {detail}"
        )
    duplicates = [m.config.name for m in loaded
                  if m.status is vr.LoadStatus.DUPLICATE]
    if duplicates:
        logger.warning("ignoring duplicate entries in PRELOAD_MODELS: %s",
                       ", ".join(duplicates))

    cpu_only = [n for n, m in models.items() if not m.on_gpu]
    if cpu_only:
        logger.warning("models running on CPU, expect very slow jobs: %s",
                       ", ".join(cpu_only))

    return models, [_describe(m) for m in loaded]


def _describe(model: vr.LoadedModel) -> dict:
    """One model, flattened to what the parent can serve from /api/health."""
    return {
        "name": model.config.name,
        "status": model.status.value,
        "device": model.device if model.ok else None,
        "load_seconds": round(model.load_seconds, 1) if model.ok else None,
        "error": model.error,
        "applied": _applied(model),
    }


def _applied(model: vr.LoadedModel) -> dict:
    """What this model is actually running with, param by param.

    This process is the only one that can answer that. A param left unset
    means "whatever the model itself says", and audio-separator resolves
    that from the checkpoint's own data while loading - so the numbers exist
    nowhere until a model is live, and only here. The settings page shows
    them instead of an empty box.

    Never raises: these are read through audio-separator's instance
    attributes, and a version that moves one is worth a blank field and a
    warning, not a worker that will not start.
    """
    if not model.ok:
        return {}
    _, sep_cls = vr.config_classes_for(model.config.name)
    instance = model.separator.model_instance
    try:
        return {**model.config.read_applied(instance),
                **sep_cls().read_applied(instance)}
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("could not read the applied params for %s",
                       model.config.name, exc_info=True)
        return {}


# --------------------------------------------------------------------- loop


def _drain(settings: Settings, models: Mapping[str, vr.LoadedModel],
           commands) -> None:
    """Run queued jobs, one at a time, until told to stop.

    Commands are read between jobs and never during one: a separation is not
    interruptible, so a STOP that arrives mid-job is honoured when the job
    ends, or by the parent terminating this process, whichever comes first.
    """
    # Own connection: sqlite handles are cheap, and this one belongs to a
    # different process from the request handlers' now, not just a different
    # thread. WAL mode is what makes that sound.
    conn = db.connect(settings.db_path)
    try:
        while True:
            job = jobs_repo.next_queued(conn)
            if job is None:
                if _stopping(commands, _IDLE_POLL_SECONDS):
                    return
                continue
            _run_one(conn, job.id, settings, models)
            if _stopping(commands, 0):
                return
    finally:
        conn.close()
        logger.info("worker loop exited")


def _stopping(commands, timeout: float) -> bool:
    """Wait up to `timeout` for the parent, and say whether to stop.

    Doubles as the idle sleep: WAKE only has to arrive to end the wait, which
    is why nothing is done with it. A closed pipe means the web process has
    gone, and there is no reason to keep two models resident for it.
    """
    try:
        if not commands.poll(timeout):
            return False
        stop = False
        while True:
            if commands.recv() == STOP:
                stop = True
            if not commands.poll(0):
                return stop
    except EOFError:
        logger.warning("the web process has gone; stopping")
        return True
    except OSError:
        logger.warning("lost the pipe to the web process; stopping",
                       exc_info=True)
        return True


def _run_one(conn, job_id: str, settings: Settings,
             models: Mapping[str, vr.LoadedModel]) -> None:
    # Built per job rather than held: the web process owns the login, so the
    # session file is where a sign-in, a token refresh or a logout shows up.
    # Rereading it here is what makes those reach the worker with no IPC, and
    # it costs one HTTP round trip against a job measured in minutes.
    client = td.TidalClient(config_dir=settings.state_dir / "tidal")
    try:
        done = pipeline.run_job(conn, job_id, client, models, settings)
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


# ---------------------------------------------------------------------- ipc


def _send(report, kind: str, payload) -> None:
    """Send one message up the pipe, tolerating a parent that has gone.

    A dead parent is not worth an exception here: the loop is about to end
    either way, and there is nobody left to read the traceback.
    """
    try:
        report.send((kind, payload))
    except (BrokenPipeError, EOFError, OSError):
        logger.warning("parent process is gone; could not report %s", kind)
