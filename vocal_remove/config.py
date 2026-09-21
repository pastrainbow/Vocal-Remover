"""Configuration for model setup and separation.

Both hierarchies are abstract with one concrete subclass per UVR architecture,
because audio-separator takes a *different parameter dict* for each
(mdxc_params, mdx_params, vr_params, demucs_params) and the keys do not
overlap. A single flat config would have to accept every key and silently drop
the ones that do not apply to the chosen model.

The split between ModelConfig and SeparationConfig is dictated by when
audio-separator reads each value:

  * load_model() copies its arch dict into the model instance, so anything
    only read there is fixed until the model is reloaded -> ModelConfig.
  * demix() reads some attributes off the instance at inference time, so those
    can be changed per job by mutating it -> SeparationConfig.

Verified by reading audio-separator 0.47.0 and by measurement: overlap and
batch_size are read inside demix() on both MDXC and MDX. segment_size is read
in demix() by MDXC but ALSO branched on in MDX's load_model()
(`self.segment_size == self.dim_t`) and used to derive chunk_size, so it is
load-time for the API as a whole.

VR and Demucs parameters are all treated as load-time: their per-inference
mutability has not been verified here.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from . import errors

_ROOT = Path(__file__).resolve().parents[1]

#: BS-Roformer (MDXC). Best quality measured: 1.8x realtime on an RTX 4060.
DEFAULT_MDXC_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
#: MDX-Net (ONNX). Measured 11.7x realtime - ~6.5x faster, lower quality.
DEFAULT_MDX_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
DEFAULT_VR_MODEL = "1_HP-UVR.pth"
DEFAULT_DEMUCS_MODEL = "htdemucs_ft.yaml"

#: What ModelConfig() used to default to, kept for callers that just want
#: "the good one" without naming an architecture.
DEFAULT_MODEL = DEFAULT_MDXC_MODEL


# --------------------------------------------------------------------- model


@dataclass(frozen=True)
class ModelConfig(ABC):
    """How a separation model is built. Changing any of this needs a reload."""

    name: str = ""
    model_dir: Path = field(default_factory=lambda: _ROOT / "data" / "models")

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

    @property
    @abstractmethod
    def arch_key(self) -> str:
        """Which Separator(...) keyword this config's params belong to."""

    @abstractmethod
    def arch_params(self) -> Dict[str, Any]:
        """Architecture parameter dict, starting from library defaults."""


@dataclass(frozen=True)
class MDXCModelConfig(ModelConfig):
    """MDXC: BS-Roformer and MDX23C checkpoints (.ckpt / .yaml)."""

    name: str = DEFAULT_MDXC_MODEL

    #: Override only for CUDA OOM. BS-Roformer is length-sensitive, so moving
    #: off its trained segment length trades separation quality for memory.
    segment_size: Optional[int] = None
    pitch_shift: int = 0

    @property
    def arch_key(self) -> str:
        return "mdxc_params"

    def arch_params(self) -> Dict[str, Any]:
        params = {"segment_size": 256, "override_model_segment_size": False,
                  "batch_size": None, "overlap": None,
                  "pitch_shift": self.pitch_shift}
        if self.segment_size is not None:
            params["segment_size"] = self.segment_size
            # Without this flag MDXC accepts segment_size then ignores it.
            params["override_model_segment_size"] = True
        return params


@dataclass(frozen=True)
class MDXModelConfig(ModelConfig):
    """MDX-Net: ONNX models (.onnx)."""

    name: str = DEFAULT_MDX_MODEL

    #: Load-time here specifically: MDX's load_model() branches on
    #: `segment_size == dim_t` and derives chunk_size from it.
    segment_size: Optional[int] = None
    hop_length: int = 1024
    enable_denoise: bool = False

    @property
    def arch_key(self) -> str:
        return "mdx_params"

    def arch_params(self) -> Dict[str, Any]:
        params = {"hop_length": self.hop_length, "segment_size": 256,
                  "overlap": 0.25, "batch_size": 1,
                  "enable_denoise": self.enable_denoise}
        if self.segment_size is not None:
            params["segment_size"] = self.segment_size
        return params


@dataclass(frozen=True)
class VRModelConfig(ModelConfig):
    """VR architecture models (.pth)."""

    name: str = DEFAULT_VR_MODEL
    window_size: int = 512
    aggression: int = 5
    batch_size: int = 1
    enable_tta: bool = False
    enable_post_process: bool = False
    post_process_threshold: float = 0.2
    high_end_process: bool = False

    @property
    def arch_key(self) -> str:
        return "vr_params"

    def arch_params(self) -> Dict[str, Any]:
        return {"batch_size": self.batch_size, "window_size": self.window_size,
                "aggression": self.aggression, "enable_tta": self.enable_tta,
                "enable_post_process": self.enable_post_process,
                "post_process_threshold": self.post_process_threshold,
                "high_end_process": self.high_end_process}


@dataclass(frozen=True)
class DemucsModelConfig(ModelConfig):
    """Demucs models (.yaml). Produce four stems, not two."""

    name: str = DEFAULT_DEMUCS_MODEL
    segment_size: str = "Default"
    shifts: int = 2
    overlap: float = 0.25
    segments_enabled: bool = True

    @property
    def arch_key(self) -> str:
        return "demucs_params"

    def arch_params(self) -> Dict[str, Any]:
        return {"segment_size": self.segment_size, "shifts": self.shifts,
                "overlap": self.overlap,
                "segments_enabled": self.segments_enabled}


# ---------------------------------------------------------------- separation


@dataclass(frozen=True)
class SeparationConfig(ABC):
    """Per-job inference settings. None of this requires a model reload."""

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
        self._apply_arch(instance)

    @abstractmethod
    def _apply_arch(self, instance) -> None:
        """Apply architecture-specific per-job overrides."""


@dataclass(frozen=True)
class MDXCSeparationConfig(SeparationConfig):
    """Per-job settings for MDXC models.

    NOTE the overlap units differ from MDXSeparationConfig - see below. This
    is the main reason these are separate classes rather than one config with
    a shared `overlap` field.
    """

    #: An integer DIVISOR: demix() computes `step = chunk_size // overlap`, so
    #: this is how many overlapping windows cover each sample. Higher is
    #: slower and usually cleaner at seams; measured on BS-Roformer,
    #: overlap 2 -> 4.9s and overlap 8 -> 6.1s on the same 12s clip, with no
    #: reload between. Defaults to the model's own value (4-8).
    overlap: Optional[int] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.overlap is not None and self.overlap < 1:
            raise errors.InvalidConfig(
                f"MDXC overlap must be an integer >= 1 (it divides the chunk "
                f"size), got {self.overlap!r}. Note MDX uses a 0-1 fraction "
                f"instead - the two are not interchangeable."
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise errors.InvalidConfig(
                f"batch_size must be >= 1, got {self.batch_size!r}")

    def _apply_arch(self, instance) -> None:
        if self.overlap is not None:
            instance.overlap = self.overlap
        if self.batch_size is not None:
            instance.batch_size = self.batch_size


@dataclass(frozen=True)
class MDXSeparationConfig(SeparationConfig):
    """Per-job settings for MDX-Net models.

    NOTE the overlap units differ from MDXCSeparationConfig - see below.
    """

    #: A FRACTION in [0, 1): demix() computes `step = int((1 - overlap) *
    #: chunk_size)`. An MDXC-style integer here makes step negative and the
    #: output non-finite, so it is rejected at construction. Library default
    #: is 0.25.
    overlap: Optional[float] = None
    batch_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()
        if self.overlap is not None and not 0.0 <= self.overlap < 1.0:
            raise errors.InvalidConfig(
                f"MDX overlap must be a fraction in [0, 1), got "
                f"{self.overlap!r}. Note MDXC uses an integer divisor instead "
                f"- the two are not interchangeable."
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise errors.InvalidConfig(
                f"batch_size must be >= 1, got {self.batch_size!r}")

    def _apply_arch(self, instance) -> None:
        if self.overlap is not None:
            instance.overlap = self.overlap
        if self.batch_size is not None:
            instance.batch_size = self.batch_size


@dataclass(frozen=True)
class VRSeparationConfig(SeparationConfig):
    """Per-job settings for VR models.

    VR's parameters have not been verified as safe to mutate after load, so
    none are exposed here; set them on VRModelConfig instead.
    """

    def _apply_arch(self, instance) -> None:
        return None


@dataclass(frozen=True)
class DemucsSeparationConfig(SeparationConfig):
    """Per-job settings for Demucs models.

    As with VR, Demucs parameters are treated as load-time only.
    """

    def _apply_arch(self, instance) -> None:
        return None
