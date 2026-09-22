#!/usr/bin/env bash
#
# Start the app, installing and initialising whatever is missing first.
#
#   ./run.sh
#
# The only entry point, and it takes no arguments. Every run is the same
# sequence, and everything in setup/ is a helper it calls:
#
#   1. Windows prerequisites, each installed if it is missing: uv, ffmpeg, a
#      CPython 3.11                                       setup/bootstrap.ps1
#   2. the venv, and the packages from requirements/app.lock, via uv
#   3. the environment check, which stops the run        setup/verify_env.py
#   4. the UVR model cache, ~0.7 GiB                    setup/fetch_models.py
#   5. the server
#
# There are no flags because there is no step worth skipping: each one is a
# no-op on a machine that is already set up, and a flag only ever encoded "I
# am fairly sure this part is fine". Being wrong about that is what produces
# a box nobody can explain, so the sequence is not negotiable.
#
# Packages come from requirements/app.lock via uv. A missing venv, a
# half-installed one, or a lockfile newer than the last install all end in the
# same place, so you do not have to think about which. Delete
# venv/.installed-from to force a reinstall on the next run.
#
# The same goes for the tools underneath: a missing one is installed rather
# than reported. Nothing here stops to tell you to go and run an installer
# first; only an install that fails is fatal.
#
# Windows: run it from Git Bash, or from PowerShell with
#   & "C:\Program Files\Git\bin\bash.exe" ./run.sh
# It uses the venv's python directly rather than "activating" anything, so
# nothing leaks into your shell.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv"
LOCK="$ROOT/requirements/app.lock"
#: Written after a successful install, so a changed lockfile is noticed.
STAMP="$VENV/.installed-from"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# Stop rather than ignore: an argument here is a habit from an older copy of
# this script, and silently dropping it would mean silently not doing what
# the caller asked for.
[ $# -eq 0 ] || die "run.sh takes no arguments (got: $*) - see the top of this file"

#: Whether the packages need (re)installing. Decided below, from the state of
#: the venv rather than from a flag.
sync_mode=no

# The venv layout differs by platform; this script has to work on both.
venv_python() {
  if [ -x "$VENV/Scripts/python.exe" ]; then echo "$VENV/Scripts/python.exe"
  elif [ -x "$VENV/bin/python" ];     then echo "$VENV/bin/python"
  fi
}

# ---------------------------------------------------------------- the tools

#: Set by run_bootstrap, so a heal later in the run does not repeat it.
bootstrap_ran=no

have_bootstrap() {
  command -v powershell.exe >/dev/null 2>&1 && [ -f "$ROOT/setup/bootstrap.ps1" ]
}

# The Windows half of "install what is missing": uv, ffmpeg, and a CPython
# 3.11 to build the venv from. Its output is for you to read; the INTERPRETER=
# and PATH_ADD= lines are for this script. \r has to go - the helper is
# PowerShell, and its line endings are CRLF. Given a python as $1 it uses that
# one and skips interpreter discovery, which is all a later heal needs.
# Returns non-zero if it did not reach the end, so each caller decides how bad
# that is.
run_bootstrap() {
  local ps1 args=() dir found
  ps1="$(cygpath -w "$ROOT/setup/bootstrap.ps1" 2>/dev/null || echo "$ROOT/setup/bootstrap.ps1")"
  if [ $# -gt 0 ]; then
    args=(-Python "$(cygpath -w "$1" 2>/dev/null || echo "$1")")
  fi

  bootstrap_log="$(mktemp)"
  trap 'rm -f "$bootstrap_log"' EXIT
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ps1" "${args[@]}" \
    | tee "$bootstrap_log" | tr -d '\r' | grep -vE '^(INTERPRETER|PATH_ADD)=' || true

  # What it installed is on PATH for the next shell, not for this one: the
  # installers edit the registry, and our copy of PATH was inherited before
  # they did. Every directory it added comes back as a PATH_ADD= line.
  while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    dir="$(cygpath -u "$dir" 2>/dev/null || echo "$dir")"
    case ":$PATH:" in *":$dir:"*) ;; *) PATH="$dir:$PATH" ;; esac
  done < <(tr -d '\r' < "$bootstrap_log" | sed -n 's/^PATH_ADD=//p')
  export PATH

  found="$(tr -d '\r' < "$bootstrap_log" | sed -n 's/^INTERPRETER=//p' | tail -1)"
  rm -f "$bootstrap_log"; trap - EXIT

  [ -n "$found" ] || return 1
  interpreter="$found"
  bootstrap_ran=yes
}

#: Where the two installers leave uv, in the order they consult.
uv_from_known_dirs() {
  local dir
  for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -n "$dir" ] || continue
    if [ -x "$dir/uv" ] || [ -x "$dir/uv.exe" ]; then
      case ":$PATH:" in *":$dir:"*) ;; *) PATH="$dir:$PATH"; export PATH ;; esac
      command -v uv >/dev/null 2>&1 && return 0
    fi
  done
  return 1
}

# uv builds the venv and installs every package into it, so it is the one tool
# this script cannot go on without. On Windows bootstrap.ps1 has usually put
# it there already; otherwise fetch it the way the docs tell you to.
ensure_uv() {
  command -v uv >/dev/null 2>&1 && return 0
  uv_from_known_dirs && return 0

  say "uv is not installed - installing it from astral.sh"
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
      -Command "irm https://astral.sh/uv/install.ps1 | iex" || true
  elif command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh || true
  else
    die "no curl, wget or powershell to install uv with - see https://docs.astral.sh/uv/"
  fi

  # The installer appends its directory to your shell profile, which does
  # nothing for the shell already running this.
  uv_from_known_dirs || die \
    "uv installed, but no uv on PATH afterwards - see https://docs.astral.sh/uv/"
  say "uv $(uv --version | awk '{print $2}') installed"
}

# ------------------------------------------------------------- prerequisites

[ -f "$LOCK" ] || die "no lockfile at $LOCK"

if [ -z "$(venv_python)" ]; then
  interpreter=3.11
  # First run on Windows: the helper installs the tools that live outside the
  # venv, then finds or installs an interpreter to build it from.
  if have_bootstrap; then
    run_bootstrap || die "setup/bootstrap.ps1 did not finish - see above"
  fi
  ensure_uv

  say "creating venv (python $interpreter)"
  uv venv "$VENV" --python "$interpreter" --allow-existing
  sync_mode=force
fi

ensure_uv

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
# It is a one-off install, so hand it to the helper rather than leave you
# with a warning and a download button that fails. The check below is what
# decides whether a still-missing one is fatal.
if ! command -v ffmpeg >/dev/null 2>&1; then
  if [ "$bootstrap_ran" = no ] && have_bootstrap; then
    run_bootstrap "$PY" || true
  fi
  command -v ffmpeg >/dev/null 2>&1 || warn \
    "ffmpeg not on PATH - downloads and separation will fail"
fi

# ------------------------------------------------------------------- check

# Behaviour, not presence: verify_env.py builds a real ONNX Runtime session
# rather than trusting get_available_providers(), which reports CUDA even
# when CUDA cannot initialise. That is the case worth stopping for - a
# CPU-only torch or a dead CUDA provider starts perfectly happily, answers
# /api/health, and runs every separation about 10x slower.
say "checking the environment"
"$PY" "$ROOT/setup/verify_env.py" || die "environment check failed - see above"

# ------------------------------------------------------------------ models

# Before the server, never inside it.
#
# A model named in preload_models but absent from data/models is not an
# error - audio-separator downloads it on first load. But that download would
# then happen while whoever started this is counting down a health check, and
# audio-separator streams straight to the final path guarded only by
# isfile(), so an interrupted download leaves a truncated file that every
# later run treats as cached. The model then never loads again until someone
# deletes it by hand.
#
# Here the download has no deadline and no process waiting to be killed, and
# fetch_models.py removes its own partial files if it fails anyway. A failure
# is therefore only a slower first job, not a reason to refuse to start.
say "caching UVR models"
"$PY" "$ROOT/setup/fetch_models.py" \
  || warn "model prefetch failed - they will download on first use instead"

# ----------------------------------------------------------------- the app

# Run from src/ so the three packages import without any PYTHONPATH, an
# editable install, or a .pth file in the venv. Everything the app reads or
# writes - data/, state/, .env - is resolved from __file__ against the repo
# root, so the working directory does not matter to it.
say "starting the server (ctrl-c to stop)"
cd "$ROOT/src"
exec "$PY" -m app.main
