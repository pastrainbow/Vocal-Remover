"""Application settings, read from environment / .env.

Per-machine values live here rather than in code because the laptop (8 GiB)
and the desktop (12 GiB) differ on what they can hold resident. Copy
.env.example to .env and edit.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The repo root: src/app/config.py -> src/app -> src -> here. data/ and
#: state/ live beside src/, not inside it.
_ROOT = Path(__file__).resolve().parents[2]

#: Shared by both processes. The worker is spawned, not forked, so it starts
#: with logging unconfigured and has to set this up for itself - keeping the
#: format in one place is what stops the two halves of one terminal looking
#: like two different programs.
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

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

    # --------------------------------------------------------------- models
    #: Preloaded at startup and held resident. No lazy loading: if these do
    #: not fit in VRAM together, startup fails loudly rather than degrading.
    preload_models: List[str] = Field(default_factory=lambda: [
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "UVR-MDX-NET-Voc_FT.onnx",
    ])
    default_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    output_format: str = "FLAC"

    #: Only set if you hit CUDA OOM; trades separation quality for memory.
    segment_size: Optional[int] = None

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
