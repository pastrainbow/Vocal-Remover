"""HTTP routes.

Submission is the only endpoint with real logic, and most of it is refusing to
create work that already exists - see submit_job().
"""
import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import tidal_download as td

from . import db, jobs as jobs_repo
from .config import Settings
from .worker import Worker

logger = logging.getLogger("app.api")

router = APIRouter(prefix="/api")

#: How often the SSE stream re-reads a job. Jobs change on the order of
#: seconds, so this is responsive without hammering SQLite.
_SSE_POLL_SECONDS = 0.5
#: Stop an abandoned stream from polling forever if a client vanishes.
_SSE_MAX_SECONDS = 60 * 60


# ---------------------------------------------------------------- plumbing


def get_worker(request: Request) -> Worker:
    return request.app.state.worker


def get_login(request: Request) -> td.LoginFlow:
    """The worker's Tidal session drives the login, so signing in from the
    page is all the worker needs - there is nothing to hand over."""
    worker = request.app.state.worker
    if worker.client is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "worker is not running; restart the server")
    return worker.client.login


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = db.connect(request.app.state.settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


# ------------------------------------------------------------------ schemas


class SubmitRequest(BaseModel):
    url: str = Field(..., description="Tidal track URL, or a bare track id")
    model: Optional[str] = None
    output_format: Optional[str] = None


# ------------------------------------------------------------------- health


@router.get("/health")
def health(worker: Worker = Depends(get_worker),
           settings: Settings = Depends(get_settings_dep)):
    """Everything a UI needs to decide what to show before accepting input."""
    try:
        auth = worker.client.auth_status() if worker.client else None
        authenticated = bool(auth and auth.valid)
        auth_detail = auth.detail if auth else "worker not started"
    except Exception as exc:  # never let a health check 500
        authenticated, auth_detail = False, f"auth check failed: {exc}"

    return {
        "worker": worker.status(),
        "tidal": {
            "authenticated": authenticated,
            "detail": auth_detail,
            # Kept as a fallback for a headless box, where nobody can click
            # the button: the CLI does the same device flow in a terminal.
            "login_command": (
                "venv\\Scripts\\python.exe -m smoke_test.tidal_cli login"),
        },
        "cache": {
            "enabled": settings.caching_enabled,
            "permanent": settings.cache_is_permanent,
            "ttl_seconds": settings.cache_ttl_seconds,
        },
        "defaults": {
            "model": settings.default_model,
            "output_format": settings.output_format,
        },
    }


@router.get("/models")
def list_models(worker: Worker = Depends(get_worker)):
    return {"models": worker.status()["models"]}


# -------------------------------------------------------------------- login


@router.post("/login", status_code=status.HTTP_202_ACCEPTED)
def start_login(flow: td.LoginFlow = Depends(get_login)):
    """Ask Tidal for a device code, and start waiting for it to be approved.

    Returns straight away with the code to show; approval is reported by
    GET /login. Asking again while one is still pending returns that same
    code rather than issuing a second one.
    """
    try:
        return flow.begin()
    except td.AuthError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"could not start a Tidal login: {exc}")


@router.get("/login")
def login_status(flow: td.LoginFlow = Depends(get_login)):
    """Where the login has got to. Cheap: it only reads state, never polls."""
    return flow.status()


@router.post("/logout")
def logout(worker: Worker = Depends(get_worker)):
    """Delete the stored Tidal session and drop the live one.

    Removes state/tidal/session.json, so signing back in means approving a
    new device code. A job already downloading will fail with an auth error -
    the session it was using has gone. Abandoning a login still in flight is
    part of logout() itself.
    """
    if worker.client is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "worker is not running; restart the server")
    worker.client.logout()
    logger.info("tidal session cleared")
    return {"authenticated": False}


# --------------------------------------------------------------------- jobs


@router.post("/jobs")
def submit_job(body: SubmitRequest, response: Response,
               worker: Worker = Depends(get_worker),
               settings: Settings = Depends(get_settings_dep),
               conn: sqlite3.Connection = Depends(get_conn)):
    """Queue a separation, or hand back work that already covers it.

    Returns 201 for a newly queued job and 200 when an existing one is
    reused. The reuse checks are not an optimisation: without them a repeat
    submission runs a full download and separation only to be rejected by the
    unique cache index at the very end.
    """
    model = body.model or settings.default_model
    output_format = (body.output_format or settings.output_format).upper()

    if not worker.running:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="worker is not running; restart the server",
        )
    if model not in worker.models:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"model {model!r} is not loaded; available: "
                   f"{sorted(worker.models)}",
        )

    # Resolve before creating anything: a bad URL should be a 400, not a job
    # that fails three seconds later.
    try:
        info = worker.client.resolve(body.url)
    except td.AuthError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    except (td.UnsupportedUrl, td.NotFound) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    except td.TidalError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc))

    key = dict(track_id=info.id, model=model, output_format=output_format)

    cached = jobs_repo.find_cached(conn, ttl_seconds=settings.cache_ttl_seconds,
                                   **key)
    if cached is not None:
        response.status_code = status.HTTP_200_OK
        return cached.to_dict()

    active = jobs_repo.find_active(conn, **key)
    if active is not None:
        response.status_code = status.HTTP_200_OK
        return active.to_dict()

    job = jobs_repo.create(
        conn, url=body.url, model=model, output_format=output_format,
        track_id=info.id, title=info.title, artist=info.artist,
        album=info.album, duration=info.duration,
    )
    worker.notify()
    response.status_code = status.HTTP_201_CREATED
    logger.info("queued job %s for %s", job.id[:8], info.display)
    return job.to_dict()


@router.get("/jobs")
def list_jobs(limit: int = 25, conn: sqlite3.Connection = Depends(get_conn)):
    return {"jobs": [j.to_dict() for j in jobs_repo.recent(conn, limit=limit)]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, conn: sqlite3.Connection = Depends(get_conn)):
    job = jobs_repo.get(conn, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    return job.to_dict()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """Server-sent events for one job, closing once it reaches a terminal stage.

    Opens its own connection: the generator outlives the request scope, so it
    cannot borrow the dependency's.
    """
    settings: Settings = request.app.state.settings

    conn = db.connect(settings.db_path)
    if jobs_repo.get(conn, job_id) is None:
        conn.close()
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")

    async def stream():
        last = None
        waited = 0.0
        try:
            while waited < _SSE_MAX_SECONDS:
                if await request.is_disconnected():
                    return
                job = jobs_repo.get(conn, job_id)
                if job is None:
                    # Evicted from the cache mid-stream.
                    yield {"event": "gone", "data": "{}"}
                    return
                payload = job.to_dict()
                if payload != last:
                    last = payload
                    yield {"event": "update", "data": json.dumps(payload)}
                if job.stage.is_terminal:
                    return
                await asyncio.sleep(_SSE_POLL_SECONDS)
                waited += _SSE_POLL_SECONDS
        finally:
            conn.close()

    return EventSourceResponse(stream())


@router.get("/jobs/{job_id}/stems/{which}")
def download_stem(job_id: str, which: str,
                  conn: sqlite3.Connection = Depends(get_conn)):
    """Serve a stem by job id.

    Resolves the path from the database rather than exposing data/ as a static
    mount, so nothing outside a completed job's own output is reachable.
    """
    if which not in ("vocals", "instrumental"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "stem must be 'vocals' or 'instrumental'")
    job = jobs_repo.get(conn, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    if job.stage is not jobs_repo.Stage.DONE:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"job is {job.stage.value}, not done")

    path_str = job.vocals_path if which == "vocals" else job.instrumental_path
    if not path_str or not Path(path_str).is_file():
        raise HTTPException(status.HTTP_410_GONE,
                            "stem file is no longer on disk")

    path = Path(path_str)
    return FileResponse(path, filename=path.name,
                        media_type="application/octet-stream")
