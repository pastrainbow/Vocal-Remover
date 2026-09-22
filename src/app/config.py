"""Application settings.

Defaults in code, with one deliberate exception: DATA_DIR and STATE_DIR are
read from the environment, because the staging deploy needs them. The Actions
runner checks out into its own workspace, so a repo-relative data/ there
would quietly build a PARALLEL installation - its own empty data/models, its
own logged-out state/tidal, its own job database. setup/deploy.ps1 sets both
to fixed paths outside the source tree to prevent that; see the comment at
the top of its param block. Every field below is in fact env-overridable,
since that is how pydantic-settings works, but those two are the only ones
anything actually sets.

There is no .env file and no support for one. The per-machine value that
used to justify it was segment_size - the laptop (8 GiB) and the desktop
(12 GiB) differ on what they can hold resident - which moved to a per-model
state file and then went entirely: every inference parameter the models
expose is either inert, fixed by the model export, or a one-way trade of
runtime for seam quality, so there was nothing left to configure. See
vocal_remove/config.py. A machine that genuinely cannot hold PRELOAD_MODELS
now loads fewer of them rather than shrinking them.

Nor is there a default model or output format here. Both used to be
fallbacks for a submission that named neither, which only ever hid a bug:
the page always sends both, so a request without them is a caller that is
wrong, and /api/jobs now rejects it instead of quietly picking something.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

#: The repo root: src/app/config.py -> src/app -> src -> here. data/ and
#: state/ live beside src/, not inside it.
_ROOT = Path(__file__).resolve().parents[2]

#: Shared by both processes. The worker is spawned, not forked, so it starts
#: with logging unconfigured and has to set this up for itself - keeping the
#: format in one place is what stops the two halves of one terminal looking
#: like two different programs.
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"

#: Preloaded at startup and held resident, in this order - the first is what
#: the page offers first. No lazy loading: if these do not fit in VRAM
#: together, startup fails loudly rather than degrading.
#:
#: BS-Roformer is the highest-SDR vocal model audio-separator ships and
#: measures 1.8x realtime on an RTX 4060; the MDX-Net one measures 11.7x at
#: lower quality. That ~6.5x spread is the only quality/speed dial this app
#: has, which is why it is a per-job choice on the page rather than a
#: setting - see vocal_remove/config.py for what the alternatives were.
#:
#: A constant rather than a setting, and the ONLY list of its kind: the
#: worker loads exactly this (see vocal_remove_worker/process.py) and
#: setup/fetch_models.py pre-downloads exactly this, so the models on disk
#: and the models loaded cannot drift apart. Deriving it from data/models
#: instead would not work - audio-separator writes its own files in there,
#: including a .yaml CONFIG beside the BS-Roformer .ckpt that dispatches as
#: an MDXC MODEL and would fail the load.
PRELOAD_MODELS = [
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "UVR-MDX-NET-Voc_FT.onnx",
]


class Settings(BaseSettings):
    # ---------------------------------------------------------------- paths
    data_dir: Path = _ROOT / "data"
    state_dir: Path = _ROOT / "state"

    # ---------------------------------------------------------------- cache
    #: Unset  -> permanent, a completed job is reused forever (default).
    #: N > 0   -> a completed job is reused for N seconds.
    #: 0       -> caching disabled, every submission does the work.
    #:
    #: Keyed on (track_id, model, output_format). The only unsoundness is that
    #: Quality.BEST resolves at download time, so a cached result could in
    #: principle have been fetched at a different quality than a fresh run
    #: would get - unlikely, and not worth re-downloading to avoid.
    cache_ttl_seconds: Optional[int] = None

    # ----------------------------------------------------------------- http
    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("cache_ttl_seconds", mode="before")
    @classmethod
    def _blank_ttl_is_permanent(cls, v):
        """An empty env var means permanent, not a parse error."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    @property
    def db_path(self) -> Path:
        return self.state_dir / "app.db"

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def caching_enabled(self) -> bool:
        return self.cache_ttl_seconds != 0

    @property
    def cache_is_permanent(self) -> bool:
        return self.cache_ttl_seconds is None

    def ensure_dirs(self) -> None:
        for d in (self.state_dir, self.staging_dir, self.out_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
