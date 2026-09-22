"""FastAPI application.

    ./run.sh                                      installs what is missing

or, if you would rather drive it yourself, from src/ so that the packages
beside this one are importable:

    ../venv/Scripts/python.exe -m app.main
    ../venv/Scripts/uvicorn.exe app.main:app --reload

Separation runs in a child process this one spawns and supervises, so a crash
in the separator costs the job that was running and not the server - see
vocal_remove_worker/. Models load during that child's startup, which this
waits for, so the first request meets a healthy worker rather than racing it.
A model that cannot load aborts startup outright.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import tidal_download as td

from .api import router
from .config import LOG_DATEFMT, LOG_FORMAT, get_settings
from .vocal_remove_worker import Worker

logger = logging.getLogger("app")

_STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    app.state.settings = settings

    # The Tidal session belongs to this process, not the worker's: the API
    # needs it synchronously to resolve a URL before creating a job and to
    # run the device login, neither of which can wait on a queue. What the
    # worker needs, it gets from the session file this writes.
    app.state.tidal = td.TidalClient(config_dir=settings.state_dir / "tidal")

    worker = Worker(settings)
    app.state.worker = worker
    # Deliberately not guarded: if the models will not load there is nothing
    # useful this server can do, and starting anyway would only turn a clear
    # startup error into a stream of failed jobs. Crashes AFTER this point
    # are a different matter - the supervisor restarts them.
    worker.start()

    try:
        yield
    finally:
        worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vocal Remover",
        description="Tidal track in, vocal and instrumental stems out.",
        lifespan=lifespan,
    )
    app.include_router(router)

    if _STATIC.is_dir():
        # Mounted last so it cannot shadow /api. Serves only app/static -
        # data/ is never exposed; stems go through a route that resolves
        # paths from the database.
        app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
