"""vocal_remove - GPU stem separation over Ultimate Vocal Remover models.

    from vocal_remove import (init_model, separate,
                              MDXCModelConfig, MDXCSeparationConfig)

    models = init_models([MDXCModelConfig()])       # a list in, a list out
    model = models[0]
    if model.ok:
        job = separate("song.flac", model,
                       MDXCSeparationConfig(output_dir="out/job123"))
        while not job.wait(0.5):                    # separate() returns at once
            print(job.get_progress())               # 0.0 -> 1.0
        stems = job.result()  # blocks if still running; raises if it failed
        stems.vocals          # Path to the vocal-only track
        stems.instrumental    # Path to the instrumental-only track

Configs are abstract with one concrete subclass per UVR architecture, because
audio-separator takes a different parameter dict for each. Pick the subclass
matching your model: MDXC for .ckpt BS-Roformer/MDX23C, MDX for .onnx
MDX-Net, VR for .pth, Demucs for .yaml.

Loading costs 1.6-3.3s warm and separation 20-126s depending on the model, so
hold the LoadedModel list for the lifetime of the worker.
"""
from .config import (
    DEFAULT_DEMUCS_MODEL,
    DEFAULT_MDX_MODEL,
    DEFAULT_MDXC_MODEL,
    DEFAULT_MODEL,
    DEFAULT_VR_MODEL,
    DemucsModelConfig,
    DemucsSeparationConfig,
    MDXCModelConfig,
    MDXCSeparationConfig,
    MDXModelConfig,
    MDXSeparationConfig,
    ModelConfig,
    SeparationConfig,
    VRModelConfig,
    VRSeparationConfig,
    config_classes_for,
)
from .errors import (
    AudioNotFound,
    Cancelled,
    InvalidConfig,
    ModelLoadError,
    OutOfMemory,
    SeparationError,
    VocalRemoveError,
)
from .separator import (
    LoadedModel,
    LoadStatus,
    Separation,
    SeparateResult,
    init_models,
    separate,
)

from . import errors  # noqa: F401  so `from vocal_remove import *` binds it

__all__ = [
    # api
    "errors",
    "init_models",
    "separate",
    "LoadedModel",
    "LoadStatus",
    "Separation",
    "SeparateResult",
    # config - abstract
    "ModelConfig",
    "SeparationConfig",
    "config_classes_for",
    # config - per architecture
    "MDXCModelConfig",
    "MDXCSeparationConfig",
    "MDXModelConfig",
    "MDXSeparationConfig",
    "VRModelConfig",
    "VRSeparationConfig",
    "DemucsModelConfig",
    "DemucsSeparationConfig",
    # defaults
    "DEFAULT_MODEL",
    "DEFAULT_MDXC_MODEL",
    "DEFAULT_MDX_MODEL",
    "DEFAULT_VR_MODEL",
    "DEFAULT_DEMUCS_MODEL",
    # errors
    "VocalRemoveError",
    "InvalidConfig",
    "ModelLoadError",
    "AudioNotFound",
    "SeparationError",
    "OutOfMemory",
    "Cancelled",
]
