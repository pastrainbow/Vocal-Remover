"""FastAPI application.

    venv\\Scripts\\python.exe -m app.main
    venv\\Scripts\\uvicorn.exe app.main:app --reload

Models load during startup, so the first request waits for a healthy worker
rather than racing it. A model that cannot load aborts startup outright -
see Worker.start().
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import get_settings
from .worker import Worker

logger = logging.getLogger("app")

_STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    app.state.settings = settings

    worker = Worker(settings)
    app.state.worker = worker
    # Deliberately not guarded: if the models will not load there is nothing
    # useful this server can do, and starting anyway would only turn a clear
    # startup error into a stream of failed jobs.
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
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
