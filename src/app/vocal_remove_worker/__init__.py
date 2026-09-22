"""The separator, and the half of it that lives in the web process.

Two modules, one per process:

    supervise.py   spawns the worker, restarts it when it dies, says why
    process.py     the child itself - models, queue, one job at a time

Only supervise.py is exported. Importing process.py pulls in torch and
audio-separator, which belong in the child and nowhere else; supervise.py
reaches it from inside the function it hands to multiprocessing, so a web
process that imports this package stays light.
"""
from .supervise import Worker, WorkerStartupError

__all__ = ["Worker", "WorkerStartupError"]
