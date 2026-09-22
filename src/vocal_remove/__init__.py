"""vocal_remove - GPU stem separation over Ultimate Vocal Remover models.

    from vocal_remove import init_models, separate, ModelConfig, SeparationConfig

    models = init_models([ModelConfig(name="...ckpt",    # a list in, list out
                                      model_dir="data/models")])
    model = models[0]
    if model.ok:
        job = separate("song.flac", model,
                       SeparationConfig(output_dir="out/job123"))
        while not job.wait(0.5):                    # separate() returns at once
            print(job.get_progress())               # 0.0 -> 1.0
        stems = job.result()  # blocks if still running; raises if it failed
        stems.vocals          # Path to the vocal-only track
        stems.instrumental    # Path to the instrumental-only track

One config class each, for every architecture. audio-separator works out
which architecture a model file is from its own data, and this package no
longer overrides any of the per-architecture inference parameters - see
config.py for what used to be there and why none of it earned its place.

Loading costs 1.6-3.3s warm and separation 20-126s depending on the model, so
hold the LoadedModel list for the lifetime of the worker.

LoadedModel, init_models, separate, Separation and SeparateResult come from
.separator, which is NOT imported at module load time - see __getattr__
below. .separator imports audio_separator and torch, multiple seconds and a
large chunk of memory that a caller wanting only .config's plain dataclasses
should never pay for. `import vocal_remove.config` or `from vocal_remove
import ModelConfig` therefore stays cheap; only touching one of the six
names below pulls .separator in, on first use.
"""
from importlib import import_module

from .config import ModelConfig, SeparationConfig
from .errors import (
    AudioNotFound,
    Cancelled,
    ModelLoadError,
    OutOfMemory,
    SeparationError,
    VocalRemoveError,
)
from . import errors  # noqa: F401  so `from vocal_remove import *` binds it

#: name -> the submodule that defines it, for __getattr__ below.
_LAZY = {
    "LoadedModel": "separator",
    "LoadStatus": "separator",
    "Separation": "separator",
    "SeparateResult": "separator",
    "init_models": "separator",
    "separate": "separator",
}


def __getattr__(name):
    """PEP 562: import .separator on first use of one of its names.

    Regular attribute access never reaches here - it only runs on a miss, so
    every access after the first is a normal (fast) module attribute lookup
    via the globals() write below.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    # api
    "errors",
    "init_models",
    "separate",
    "LoadedModel",
    "LoadStatus",
    "Separation",
    "SeparateResult",
    # config
    "ModelConfig",
    "SeparationConfig",
    # errors
    "VocalRemoveError",
    "ModelLoadError",
    "AudioNotFound",
    "SeparationError",
    "OutOfMemory",
    "Cancelled",
]
