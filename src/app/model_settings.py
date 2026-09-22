"""Model configuration: which models to preload, the default, format and
segment size.

Stored in state/model_settings.json rather than .env, unlike the rest of
config.py: these four are meant to be changed at runtime from the model
settings page in the GUI, and a file the app rewrites itself is a better fit
for that than an environment file meant to be hand-edited once at deploy
time.

Deliberately stdlib-only, like db.py and jobs.py - see tests/test_jobs.py for
why that matters. app.config still imports vocal_remove for nothing (not even
its error types): importing that package pulls in audio_separator and torch
through vocal_remove/__init__.py -> separator.py, which the web process must
never pay for. Default model filenames are therefore repeated here as plain
strings, the same way app/config.py already does.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("app.model_settings")

#: Lives directly in state/, alongside app.db and tidal/ - all three are
#: per-install state, not source.
FILENAME = "model_settings.json"

VALID_OUTPUT_FORMATS = ("FLAC", "WAV", "MP3")

#: Mirrors vocal_remove.config's own defaults - see the module docstring for
#: why this does not just import them.
DEFAULT_MDXC_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_MDX_MODEL = "UVR-MDX-NET-Voc_FT.onnx"


class InvalidModelSettings(ValueError):
    """A settings value is invalid. The API turns this into a 400."""


@dataclass
class ModelSettings:
    preload_models: List[str] = field(
        default_factory=lambda: [DEFAULT_MDXC_MODEL, DEFAULT_MDX_MODEL])
    default_model: str = DEFAULT_MDXC_MODEL
    output_format: str = "FLAC"
    #: Only set if you hit CUDA OOM; trades separation quality for memory.
    segment_size: Optional[int] = None

    def validate(self) -> None:
        """Raise InvalidModelSettings if this would not be usable.

        Called before every write and after every read, so a hand-edited or
        half-written file behaves like a rejected save rather than a worker
        that will not start.
        """
        if not self.preload_models:
            raise InvalidModelSettings("preload_models must not be empty")
        if self.default_model not in self.preload_models:
            raise InvalidModelSettings(
                f"default_model {self.default_model!r} must be one of "
                f"preload_models {self.preload_models!r}")
        if self.output_format.upper() not in VALID_OUTPUT_FORMATS:
            raise InvalidModelSettings(
                f"output_format must be one of {VALID_OUTPUT_FORMATS}, got "
                f"{self.output_format!r}")
        if self.segment_size is not None and self.segment_size < 1:
            raise InvalidModelSettings(
                f"segment_size must be a positive integer, got "
                f"{self.segment_size!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def path_for(state_dir: Path) -> Path:
    return Path(state_dir) / FILENAME


def load(state_dir: Path) -> ModelSettings:
    """The persisted model settings, or the defaults if none are saved yet.

    Never raises: a missing, corrupt or invalid file falls back to defaults
    with a warning rather than taking the server down, since this runs on
    every settings read, not just at startup.
    """
    path = path_for(state_dir)
    if not path.is_file():
        return ModelSettings()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s (%s: %s); using defaults", path,
                       type(exc).__name__, exc)
        return ModelSettings()

    defaults = ModelSettings()
    settings = ModelSettings(
        preload_models=list(raw.get("preload_models", defaults.preload_models)),
        default_model=raw.get("default_model", defaults.default_model),
        output_format=raw.get("output_format", defaults.output_format),
        segment_size=raw.get("segment_size", defaults.segment_size),
    )
    try:
        settings.validate()
    except InvalidModelSettings as exc:
        logger.warning("%s has invalid settings (%s); using defaults", path, exc)
        return defaults
    return settings


def save(state_dir: Path, settings: ModelSettings) -> None:
    """Validate, then persist. Raises InvalidModelSettings without writing.

    Writes to a temp file and renames over the real one, so a crash or a
    second writer mid-save leaves either the old file or the new one, never a
    truncated one - the same concern fetch_models.py has about model
    downloads, applied to a much smaller file.
    """
    settings.validate()
    path = path_for(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
