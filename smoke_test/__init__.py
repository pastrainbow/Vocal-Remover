"""Manual dev/smoke-test scripts.

Nothing here is imported by the web app - these exist to exercise the real
libraries by hand. Run them as modules from the repo root, e.g.

    venv/Scripts/python.exe -m smoke_test.tidal_cli status
    venv/Scripts/python.exe -m smoke_test.gate_uvr

The packages they exercise live in src/, which is not on the path when you
run from the repo root, so this puts it there. It belongs here rather than in
each script: these are dev scripts run by hand from a shell, while the app
itself is started by run.sh, which runs from inside src/.
"""
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
