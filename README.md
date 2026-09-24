# Vocal Remover

A local web app that takes a Tidal track URL and gives you back two stems:
**vocals** and **instrumental**.

Paste a link like `https://tidal.com/track/242542596`, pick a model and an
output format, and the app downloads the track from Tidal at the best quality
your account can stream, separates it on the GPU, and offers both stems for
download.

## How it works

```
Tidal URL ──► download (tidalapi) ──► separate (audio-separator, CUDA) ──► vocals + instrumental
```

- **Web UI and API**: FastAPI serves a single page and a small JSON API under
  `/api`. Job progress streams to the page over Server-Sent Events.
- **Tidal download**: [`tidalapi`](https://github.com/tamland/python-tidal)
  resolves the track and fetches the highest-quality stream your subscription
  allows. You sign in once from the page with Tidal's device-code flow.
- **Separation**: [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator)
  runs UVR models in a supervised child process, so a crash during separation
  fails that one job and leaves the server running. Both models load into VRAM
  at startup and stay there.
- **Caching**: a finished job is keyed on (track, model, format). Submitting
  the same combination again returns the existing stems immediately.

### Models

You pick the model per job on the page:

| Model | Quality | Speed (RTX 4060) |
|---|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` (BS-Roformer) | Highest | ~1.8× realtime |
| `UVR-MDX-NET-Voc_FT.onnx` (MDX-Net) | Lower | ~11.7× realtime |

### Output formats

FLAC, WAV or MP3.

## Requirements

- **Windows 10/11.** The lockfile is compiled for Windows.
- **An NVIDIA GPU** with a recent driver that supports CUDA 12.8. Both models
  stay loaded at once, so budget about 8 GiB of VRAM. The app refuses to start
  on a CPU-only setup, because separating on the CPU is roughly 10× slower.
- **[Git for Windows](https://gitforwindows.org/)**, which provides Git Bash
  to run `run.sh`.
- **A Tidal subscription.** Lossless and hi-res downloads depend on your tier.
- **About 5 GB of free disk** for the CUDA wheels (~3 GB) and the model cache
  (~0.7 GiB), plus space for downloaded tracks and stems.

You don't need to install Python, uv or ffmpeg yourself. `run.sh` installs
whichever of them are missing.

## Setup

### 1. Clone

```bash
git clone <repo-url> Vocal-Remover
cd Vocal-Remover
```

### 2. Run

From Git Bash:

```bash
./run.sh
```

Or from PowerShell:

```powershell
& "C:\Program Files\Git\bin\bash.exe" ./run.sh
```

`run.sh` is the only entry point and takes no arguments. Every run goes
through the same steps, and each step does nothing if it's already done:

1. **Prerequisites** (`setup/bootstrap.ps1`): installs uv, ffmpeg (through
   winget, or a static build if winget isn't available) and CPython 3.11 if
   any of them is missing.
2. **Packages**: creates `venv/` and installs `requirements/app.lock` with
   uv. This step re-runs whenever the lockfile changes or an import fails.
3. **Environment check** (`setup/verify_env.py`): confirms that torch sees
   CUDA, that ONNX Runtime can open a CUDA session, and that ffmpeg is on
   PATH. If any check fails, the run stops here.
4. **Models** (`setup/fetch_models.py`): downloads the UVR models into
   `data/models/`. This happens once and is about 0.7 GiB.
5. **Server**: starts on <http://127.0.0.1:8000>.

The first run takes a while because of the ~3 GB of CUDA wheels. Later runs
start in under a minute.

### 3. Sign in to Tidal

Open <http://127.0.0.1:8000>. If you aren't signed in, a banner at the top of
the page offers a sign-in button. Click it, open the link it shows, and
approve the code in your Tidal account. The banner clears once the session is
live.

The session is saved under `state/tidal/`, so you only sign in once per
machine.

### 4. Separate a track

1. Paste a Tidal track URL. A bare track ID also works. Album, playlist and
   mix URLs are rejected.
2. Choose a model and an output format.
3. Click **Separate** and follow the job as it goes from queued to
   downloading, separating and done.
4. Download the vocal and instrumental stems from the job card.

Stems are also written to `data/out/<job-id>/`.

## Configuration

The defaults cover normal use. Any of these can be overridden with an
environment variable:

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `./data` | Models, downloaded tracks and stems |
| `STATE_DIR` | `./state` | Job database and the Tidal session token |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Port |
| `CACHE_TTL_SECONDS` | unset (keep forever) | How long a finished job is reused. `0` turns caching off |

`data/` and `state/` are gitignored. Keep `state/` private, because it holds
your Tidal OAuth token.

## API

The page is a thin client over these endpoints, so you can also call them
directly:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Worker status, loaded models, Tidal sign-in state |
| `GET` | `/api/models` | Models available to submit with |
| `POST` | `/api/login` | Start the Tidal device-code sign-in |
| `GET` | `/api/login` | Sign-in progress |
| `POST` | `/api/logout` | Sign out of Tidal |
| `POST` | `/api/jobs` | Submit `{"url", "model", "output_format"}`. Returns `201` for a new job, `200` when an existing one is reused |
| `GET` | `/api/jobs` | Recent jobs |
| `GET` | `/api/jobs/{id}` | One job |
| `GET` | `/api/jobs/{id}/events` | Live progress stream (SSE) |
| `GET` | `/api/jobs/{id}/stems/{vocals\|instrumental}` | Download a stem |

Interactive docs are at <http://127.0.0.1:8000/docs>.

## Project layout

```
run.sh                     the entry point: install, check, fetch models, serve
setup/                     helpers that run.sh calls, plus deploy.ps1 for staging
requirements/app.in        direct dependencies (compiled into app.lock)
src/app/                   FastAPI app, job store, pipeline, static web page
src/app/vocal_remove_worker/  supervised separation process
src/tidal_download/        Tidal sign-in, URL resolution and download
src/vocal_remove/          model loading and separation on top of audio-separator
tests/                     standard-library unit tests
```

## Development

The tests use only the standard library and don't need a GPU:

```bash
python -m unittest discover -s tests -v
```

To change dependencies, edit `requirements/app.in` and recompile the lockfile
with the command at the top of that file. On the next start, `run.sh` notices
the new lockfile and reinstalls.
