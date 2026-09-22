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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """How a separation model is built. Changing any of this needs a reload.

    `name` has no default on purpose. There is exactly one list of models
    in this project - app.config.PRELOAD_MODELS - and a default here would
    be a second, quieter one: a caller that forgot to say which model would
    silently load whichever this module happened to name.

    `model_dir` has none for the same reason. app.Settings.models_dir is the
    one place that decides it, honouring DATA_DIR; a default derived from
    this file's location would always be <source tree>/data/models, which is
    wrong wherever the source tree is not the installation (a CI runner's
    workspace, say), and a caller that forgot to pass it would load from -
    and download into - a directory nothing else reads.
    """

    name: str
    model_dir: Path

    #: Mixed precision. Roughly halves VRAM with no audible cost.
    #: Ignored by ONNX models, which run at their native precision.
    use_autocast: bool = True

    #: Regional torch.compile of the RoFormer transformer blocks: 62.7s ->
    #: 36.4s on a 120s excerpt (RTX 3060), stems within 74 dB of eager.
    #: Needs Triton - triton-windows on Windows, see requirements/app.in -
    #: and without it audio-separator falls back to eager inference. CUDA
    #: only; ignored by ONNX models.
    use_torch_compile: bool = True

    log_level: int = logging.WARNING

    def __post_init__(self):
        object.__setattr__(self, "model_dir", Path(self.model_dir))

    @property
    def key(self) -> str:
        """Identity for duplicate detection when loading a batch."""
        return self.name


@dataclass(frozen=True)
class SeparationConfig:
    """Per-job output settings. None of this requires a model reload.

    `output_dir` has no default, like ModelConfig.model_dir: where output
    goes is app.Settings' decision, and each job gets its own directory
    under it anyway.
    """

    output_dir: Path

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
