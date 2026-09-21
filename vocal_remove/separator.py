"""Model setup and stem separation.

Two public functions:

    models = init_models([MDXCModelConfig(), MDXModelConfig()])  # once
    result = separate("song.flac", models[0],
                      MDXCSeparationConfig(output_dir="out/job1"))
    result.vocals, result.instrumental                           # both Paths

init_models() is deliberately separate from separate() because loading costs
1.6-3.3s warm (and 20-30s cold, including the download) while separation takes
20-126s depending on the model. Hold the returned models for the lifetime of
the worker process rather than reloading per job.

It takes a LIST so a worker can bring up every model it might be asked for in
one pass, and it never raises for a bad entry: each result carries a status,
so one unusable model does not stop the others loading.
"""
# torch MUST be imported before onnxruntime (which audio_separator imports):
# it registers the CUDA DLL directory that ORT's provider library needs.
# Reorder these and every .onnx model silently falls back to CPU, roughly 10x
# slower, with no error anywhere. See requirements/app.in.
import torch  # noqa: F401  # isort: skip

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

from audio_separator.separator import Separator

from . import errors
from .config import ModelConfig, SeparationConfig

logger = logging.getLogger("vocal_remove")

#: Stem keys audio-separator uses for two-stem models.
_VOCALS = "Vocals"
_INSTRUMENTAL = "Instrumental"


class LoadStatus(Enum):
    """Outcome of one entry in an init_models() batch."""

    LOADED = "loaded"
    #: An earlier config in the same batch already claimed this model key.
    DUPLICATE = "duplicate"
    #: Download or load failed; see LoadedModel.error.
    FAILED = "failed"


@dataclass
class LoadedModel:
    """One entry of an init_models() batch.

    Only usable when status is LOADED; `separator` is None otherwise. Check
    `ok` before passing it to separate(), which raises on an unusable model.
    """

    config: ModelConfig
    status: LoadStatus
    separator: Optional[Separator] = None
    device: str = "cpu"
    load_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status is LoadStatus.LOADED

    @property
    def on_gpu(self) -> bool:
        return self.ok and self.device == "cuda"


@dataclass(frozen=True)
class SeparateResult:
    """Separated tracks and result metadata"""

    vocals: Path
    instrumental: Path
    model: str
    seconds: float  # wall-clock separation time

    def __iter__(self):
        """Allows `vocals, instrumental = separate(...)`."""
        return iter((self.vocals, self.instrumental))


def init_models(configs: Iterable[ModelConfig]) -> List[LoadedModel]:
    """Load each model, returning one LoadedModel per config, in order.

    Never raises for a bad entry: a repeated model key yields DUPLICATE and a
    download or load failure yields FAILED, so a single broken config cannot
    stop a worker bringing up the rest.
    """
    configs = list(configs)
    results: List[LoadedModel] = []
    seen = set()

    for config in configs:
        if config.key in seen:
            logger.warning("skipping duplicate model %r", config.key)
            results.append(LoadedModel(
                config=config, status=LoadStatus.DUPLICATE,
                error=f"{config.key!r} already loaded earlier in this batch",
            ))
            continue
        seen.add(config.key)
        results.append(_load_one(config))

    loaded = sum(1 for r in results if r.ok)
    logger.info("initialised %d/%d models", loaded, len(results))
    return results


def _load_one(config: ModelConfig) -> LoadedModel:
    config.model_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        separator = Separator(
            log_level=config.log_level,
            model_file_dir=str(config.model_dir),
            output_dir=str(config.model_dir),  # replaced per job in separate()
            use_autocast=config.use_autocast,
            **{config.arch_key: config.arch_params()},
        )
        separator.load_model(config.name)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        logger.warning("could not load %r: %s", config.name, detail)
        return LoadedModel(config=config, status=LoadStatus.FAILED, error=detail)

    device = str(getattr(separator, "torch_device", "cpu"))
    load_seconds = time.time() - started

    if not device.startswith("cuda"):
        logger.warning(
            "%r loaded on %s, not CUDA - separation will be many times "
            "slower. Check torch.cuda.is_available().", config.name, device,
        )
    logger.info("loaded %s on %s in %.1fs", config.name, device, load_seconds)

    return LoadedModel(
        config=config,
        status=LoadStatus.LOADED,
        separator=separator,
        device="cuda" if device.startswith("cuda") else device,
        load_seconds=load_seconds,
    )


def separate(
    audio_path,
    model: LoadedModel,
    config: SeparationConfig,
) -> SeparateResult:
    """Split one audio file into vocal-only and instrumental-only stems.

    Returns the two output paths. Raises ModelLoadError if the model is not
    in a usable state, plus OutOfMemory, SeparationError or AudioNotFound.
    """
    if model is None or not model.ok:
        status = model.status.value if model is not None else "missing"
        raise errors.ModelLoadError(
            f"model is not usable (status={status}): "
            f"{(model.error if model is not None else 'no model given')}"
        )

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise errors.AudioNotFound(f"no such audio file: {audio_path}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    names = {
        _VOCALS: config.vocals_name or f"{stem} (Vocals)",
        _INSTRUMENTAL: config.instrumental_name or f"{stem} (Instrumental)",
    }

    config.apply(model.separator.model_instance)
    model.separator.output_dir = str(config.output_dir)
    model.separator.output_format = config.output_format

    started = time.time()
    try:
        produced = model.separator.separate(str(audio_path), custom_output_names=names)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise errors.OutOfMemory(
            "CUDA ran out of memory. Lower segment_size on the ModelConfig "
            "(try 128) or batch_size on the SeparationConfig, then reload."
        ) from exc
    except Exception as exc:
        raise errors.SeparationError(
            f"separation failed: {type(exc).__name__}: {exc}"
        ) from exc
    elapsed = time.time() - started

    # separate() returns bare filenames, not paths, and their order is not
    # guaranteed - match on the names we asked for rather than position.
    resolved = {}
    for key, requested in names.items():
        for produced_name in produced:
            if Path(produced_name).stem == requested:
                resolved[key] = config.output_dir / Path(produced_name).name
                break

    missing = [k for k in (_VOCALS, _INSTRUMENTAL) if k not in resolved]
    if missing:
        raise errors.SeparationError(
            f"{model.config.name!r} did not produce {missing} - it may not be a "
            f"two-stem vocals/instrumental model. Got: {list(produced)}"
        )
    for key, path in resolved.items():
        if not path.is_file():
            raise errors.SeparationError(f"{key} stem missing on disk: {path}")

    logger.info("separated %s in %.1fs on %s", audio_path.name, elapsed, model.device)
    return SeparateResult(
        vocals=resolved[_VOCALS],
        instrumental=resolved[_INSTRUMENTAL],
        model=model.config.name,
        seconds=elapsed,
    )
