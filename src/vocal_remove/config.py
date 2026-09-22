"""Configuration for model setup and separation.

Two plain dataclasses, split by when audio-separator reads each value:

  * load_model() fixes what the model IS until it is reloaded -> ModelConfig.
  * separate() reads the output settings off the instance per file, so those
    can change between jobs -> SeparationConfig.

Neither carries architecture parameters, and that is deliberate.
audio-separator takes a different parameter dict per architecture
(mdxc_params, mdx_params, vr_params, demucs_params), and this module used to
mirror all four with a ModelConfig/SeparationConfig subclass each so they
could be edited per model. Every field in them turned out to be inert, fixed
by the model export, or a one-directional trade of runtime for seam quality:

  * batch_size did nothing on either preloaded model. BS-Roformer skips
    batching outright ("not utilized due to negligible performance
    improvements", mdxc_separator.py), and MDX's demix() splits a tensor
    whose batch dimension is always 1 - the residue of an older code path
    whose other half, initialize_mix(), is now called from nowhere.
  * MDX's segment_size has exactly one correct value: load_model() takes the
    ONNX Runtime path only while it equals the model's own dim_t, and falls
    back to an onnx2torch conversion ("processing may be slower") otherwise.
    hop_length is likewise the stride the model was exported with.
  * MDXC's segment_size defaults to the checkpoint's TRAINED chunk length
    (BS-Roformer: dim_t 801, ie 441 * 800 = 352800 samples = 8.0s, matching
    its own audio.chunk_size), so it is the quality optimum rather than a
    memory-conservative guess. It survived only as a CUDA OOM escape hatch.
  * overlap and enable_denoise buy marginally smoother seams with runtime,
    monotonically, so there is no optimum to search for.

So this package now passes audio-separator NO architecture parameters and
lets it use its own defaults, which are identical field for field to the
dicts this module used to build - compare Separator.__init__. Nothing about
the separation changed when they went.

If a parameter ever needs tuning again, reach for audio-separator's own
kwargs at the call site in separator.py rather than rebuilding a settings
surface here.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: The repo root: src/vocal_remove/config.py -> src/vocal_remove -> src ->
#: here. Models and output land in data/, which sits beside src/.
_ROOT = Path(__file__).resolve().parents[2]

#: Where models are downloaded to and loaded from, unless a caller says
#: otherwise. Public because setup/fetch_models.py needs the path without
#: having a model in hand, and deriving it a second time is how a prefetch
#: ends up filling a directory the server never reads.
DEFAULT_MODEL_DIR = _ROOT / "data" / "models"


@dataclass(frozen=True)
class ModelConfig:
    """How a separation model is built. Changing any of this needs a reload.

    `name` has no default on purpose. There is exactly one list of models
    in this project - app.config.PRELOAD_MODELS - and a default here would
    be a second, quieter one: a caller that forgot to say which model would
    silently load whichever this module happened to name.
    """

    name: str
    model_dir: Path = field(default_factory=lambda: DEFAULT_MODEL_DIR)

    #: Mixed precision. Roughly halves VRAM with no audible cost.
    #: Ignored by ONNX models, which run at their native precision.
    use_autocast: bool = True

    log_level: int = logging.WARNING

    def __post_init__(self):
        object.__setattr__(self, "model_dir", Path(self.model_dir))

    @property
    def key(self) -> str:
        """Identity for duplicate detection when loading a batch."""
        return self.name


@dataclass(frozen=True)
class SeparationConfig:
    """Per-job output settings. None of this requires a model reload."""

    output_dir: Path = field(default_factory=lambda: _ROOT / "data" / "out")

    #: WAV is lossless and instant to write; FLAC is lossless and ~40% smaller
    #: but costs encode time. Separation output is float32 internally, so
    #: neither loses anything the model produced.
    output_format: str = "FLAC"

    #: Stem filenames without extension. None derives them from the input.
    vocals_name: Optional[str] = None
    instrumental_name: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def apply(self, instance) -> None:
        """Mutate a loaded model instance for this job.

        load_model() copies these onto the instance and separate() reads only
        the instance, so setting them on the Separator alone has no effect.
        """
        instance.output_dir = str(self.output_dir)
        instance.output_format = self.output_format
