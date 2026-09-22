#!/usr/bin/env bash
#
# Start the app, installing whatever is missing first.
#
#   ./run.sh              start it
#   ./run.sh --sync       reinstall from the lockfile first, then start
#   ./run.sh --check      install if needed, verify the environment, exit
#   ./run.sh --models     pre-download the UVR models (~0.7 GiB), then start
#
# The only entry point. Everything in setup/ is a helper this calls:
#
#   setup/bootstrap.ps1   Windows prerequisites, on a first run: uv, ffmpeg,
#                         a CPython 3.11. Prints the interpreter to use.
#   setup/verify_env.py   checks that each layer actually works
#   setup/fetch_models.py warms the model cache for --models
#
# Packages come from requirements/app.lock via uv. A missing venv, a
# half-installed one, or a lockfile newer than the last install all end in the
# same place, so you do not have to think about which.
#
# Windows: run it from Git Bash. It uses the venv's python directly rather
# than "activating" anything, so nothing leaks into your shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
LOCK="$ROOT/requirements/app.lock"
#: Written after a successful install, so a changed lockfile is noticed.
STAMP="$VENV/.installed-from"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

sync_mode=no
check_only=no
get_models=no
for arg in "$@"; do
  case "$arg" in
    --sync)   sync_mode=force ;;
    --check)  check_only=yes ;;
    --models) get_models=yes ;;
    # The usage block at the top of this file, minus the comment markers.
    -h|--help) sed -n '3,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $arg (try --help)" ;;
  esac
done

# The venv layout differs by platform; this script has to work on both.
venv_python() {
  if [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"
  elif [ -x "$VENV/bin/python" ];     then echo "$VENV/bin/python"
  fi
}

# ------------------------------------------------------------- prerequisites

[ -f "$LOCK" ] || die "no lockfile at $LOCK"

if [ -z "$(venv_python)" ]; then
  interpreter=3.11
  if command -v powershell.exe >/dev/null 2>&1 \
     && [ -f "$ROOT/setup/bootstrap.ps1" ]; then
    # First run on Windows: let the helper find or install an interpreter and
    # check the tools that live outside the venv. Its output is for you to
    # read; the INTERPRETER= line is for this script. \r has to go - the
    # helper is PowerShell, and its line endings are CRLF.
    bootstrap_log="$(mktemp)"
    trap 'rm -f "$bootstrap_log"' EXIT
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -File "$(cygpath -w "$ROOT/setup/bootstrap.ps1" 2>/dev/null || echo "$ROOT/setup/bootstrap.ps1")" \
      | tee "$bootstrap_log" | tr -d '\r' | grep -v '^INTERPRETER=' || true
    found="$(tr -d '\r' < "$bootstrap_log" | sed -n 's/^INTERPRETER=//p' | tail -1)"
    rm -f "$bootstrap_log"; trap - EXIT
    [ -n "$found" ] || die "setup/bootstrap.ps1 did not finish - see above"
    interpreter="$found"
  else
    command -v uv >/dev/null 2>&1 || die \
      "uv is not installed - see https://docs.astral.sh/uv/"
  fi

  say "creating venv (python $interpreter)"
  uv venv "$VENV" --python "$interpreter" --allow-existing
  sync_mode=force
fi

command -v uv >/dev/null 2>&1 || die \
  "uv is not installed - see https://docs.astral.sh/uv/"

PY="$(venv_python)"
[ -n "$PY" ] || die "venv at $VENV has no python"

# ---------------------------------------------------------------- packages

if [ "$sync_mode" != force ]; then
  # Cheap "is anything missing" check: import the heavyweights the app cannot
  # start without. Anything subtler is the lockfile's job, below.
  if ! "$PY" -c "import torch, onnxruntime, fastapi, uvicorn, tidalapi, audio_separator" \
       >/dev/null 2>&1; then
    say "dependencies missing or broken"
    sync_mode=force
  elif [ ! -f "$STAMP" ] || ! cmp -s "$LOCK" "$STAMP"; then
    say "lockfile changed since the last install"
    sync_mode=force
  fi
fi

if [ "$sync_mode" = force ]; then
  say "installing from $(basename "$LOCK")"
  # --index-strategy unsafe-best-match is REQUIRED at sync time, not just when
  # compiling: app.lock pins torch to a +cu128 local version that only the
  # PyTorch index carries, and uv otherwise refuses to look past PyPI. Without
  # it this fails outright; with the wrong pins it would silently install a
  # CPU-only torch, which runs everything about 10x slower and says nothing.
  VIRTUAL_ENV="$VENV" uv pip sync "$LOCK" --index-strategy unsafe-best-match
  cp "$LOCK" "$STAMP"
  say "packages installed"
fi

# ffmpeg lives outside every venv, so no lockfile can catch a missing one.
# Only decode and remux need it, so this is a warning rather than a refusal to
# start - bootstrap.ps1 treats it as fatal at setup time, which is the moment
# to fix it.
command -v ffmpeg >/dev/null 2>&1 || warn \
  "ffmpeg not on PATH - downloads and separation will fail"

if [ "$get_models" = yes ]; then
  say "caching UVR models"
  "$PY" "$ROOT/setup/fetch_models.py"
fi

if [ "$check_only" = yes ]; then
  exec "$PY" "$ROOT/setup/verify_env.py"
fi

# ----------------------------------------------------------------- the app

# Run from src/ so the three packages import without any PYTHONPATH, an
# editable install, or a .pth file in the venv. Everything the app reads or
# writes - data/, state/, .env - is resolved from __file__ against the repo
# root, so the working directory does not matter to it.
say "starting the server (ctrl-c to stop)"
cd "$ROOT/src"
exec "$PY" -m app.main
