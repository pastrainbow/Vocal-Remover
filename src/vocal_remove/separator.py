"""Model setup and stem separation.

Two public functions:

    models = init_models([MDXCModelConfig(), MDXModelConfig()])  # once
    job = separate("song.flac", models[0],
                   MDXCSeparationConfig(output_dir="out/job1"))   # returns now
    job.get_progress()                                            # 0.0 - 1.0
    result = job.result()                                         # blocks
    result.vocals, result.instrumental                            # both Paths

separate() is asynchronous because separation takes 20-126s and callers need
to show progress while it runs; see the Separation class below.

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
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

from audio_separator.separator import Separator

from . import errors, progress
from .config import ModelConfig, SeparationConfig

logger = logging.getLogger("vocal_remove")

#: Stem keys audio-separator uses for two-stem models.
_VOCALS = "Vocals"
_INSTRUMENTAL = "Instrumental"

#: How much of the 0-1 range the chunk loop owns. The rest is the tail after
#: the last chunk - inverting the second stem and writing both - which is not
#: instrumented. Measured on the 229.5s test track: 0.4s of 122.6s on
#: BS-Roformer, and 1-2s of ~19s on MDX-Net where it is disk-bound on two
#: FLACs. So 3% covers the slow model and leaves the fast one sitting at 97%
#: for a second or two. (Decoding the input, before the first chunk, sits at 0
#: instead: 3s of 19s on MDX-Net. There is nothing to count there.)
_CHUNK_SHARE = 0.97


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

    #: Serialises separations on this model. audio-separator keeps per-job
    #: settings (output dir, format, overlap) on the one shared model
    #: instance, so two concurrent separations would read each other's.
    lock: threading.Lock = field(default_factory=threading.Lock,
                                 repr=False, compare=False)

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
) -> "Separation":
    """Start splitting one audio file into vocal and instrumental stems.

    Returns immediately with a handle on the work, which runs on a background
    thread:

        job = separate("song.flac", model, MDXCSeparationConfig(...))
        while not job.wait(0.5):
            print("%.0f%%" % (job.get_progress() * 100))
        stems = job.result()          # the failure, if any, surfaces here

    This call raises nothing. Every failure - an unusable model, a missing
    file, CUDA OOM - comes out of result() instead, so callers handle errors
    in one place rather than two.
    """
    return Separation(audio_path, model, config)


class Separation:
    """A separation running on a background thread.

    Progress is real, counted off the chunk loop inside audio-separator (see
    progress.py), except on architectures whose loops are not modelled, where
    `indeterminate` is True and get_progress() only reports 0.0 or 1.0.

    Separations on the same LoadedModel are serialised - audio-separator keeps
    per-job settings on the one shared model instance - so a second one queues
    behind the first and reports no progress until it starts.
    """

    def __init__(self, audio_path, model: LoadedModel,
                 config: SeparationConfig):
        self.audio_path = Path(audio_path)
        self.model = model
        self.config = config

        self._started = time.time()
        self._finished: Optional[float] = None
        self._result: Optional[SeparateResult] = None
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        # Built up front, not in the thread, so cancel() works from the moment
        # this returns - including before the first chunk is reached.
        self._tracker = progress.tracker_for(
            getattr(model.separator, "model_instance", None)
            if model is not None else None
        )

        # A bare daemon thread, NOT ThreadPoolExecutor: the executor's atexit
        # hook joins its workers on shutdown, which would block until a
        # runaway separation finishes. A daemon thread dies with the process.
        self._thread = threading.Thread(
            target=self._run, name=f"separate-{self.audio_path.stem[:16]}",
            daemon=True,
        )
        self._thread.start()

    # --------------------------------------------------------------- status

    def get_progress(self) -> float:
        """How far along this separation is, from 0.0 to 1.0.

        1.0 means finished and successful. A separation that failed or was
        cancelled stops where it got to, so the number still says where it
        died. Never goes backwards.
        """
        if self._done.is_set() and self._error is None:
            return 1.0
        return self._chunk_progress()

    def _chunk_progress(self) -> float:
        if self._tracker.indeterminate:
            return 0.0
        return min(self._tracker.fraction * _CHUNK_SHARE, _CHUNK_SHARE)

    @property
    def indeterminate(self) -> bool:
        """True when this architecture reports no usable chunk counts."""
        return self._tracker.indeterminate

    @property
    def done(self) -> bool:
        """True once the work has finished, failed or been cancelled."""
        return self._done.is_set()

    @property
    def error(self) -> Optional[BaseException]:
        """How it failed, without raising. None while running or on success."""
        return self._error

    @property
    def seconds(self) -> float:
        """Wall clock since the start, frozen once finished."""
        return (self._finished or time.time()) - self._started

    # ---------------------------------------------------------------- await

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block for up to `timeout`. True if the separation has finished."""
        return self._done.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> SeparateResult:
        """The stems, waiting for them if necessary.

        Raises whatever the separation raised - ModelLoadError, AudioNotFound,
        OutOfMemory, SeparationError, or Cancelled after a cancel() - and
        TimeoutError if `timeout` passes with the work still running, which
        leaves it running.
        """
        if not self._done.wait(timeout):
            raise TimeoutError(
                f"separation of {self.audio_path.name} is still running after "
                f"{timeout}s"
            )
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self) -> None:
        """Ask the separation to stop, and return without waiting.

        It stops at the next chunk boundary, so within a chunk's worth of work
        (under a second on the measured models) once inference has started.
        Cancelling before then - while the input is decoding - takes effect at
        the first chunk instead. result() then raises Cancelled.

        Best effort: Demucs is not instrumented and ignores this, and neither
        is writing the stems, so a cancel after the last chunk may still
        produce a result.
        """
        self._tracker.cancel()

    # ----------------------------------------------------------------- work

    def _run(self) -> None:
        try:
            if self.model is None or not self.model.ok:
                status = (self.model.status.value
                          if self.model is not None else "missing")
                raise errors.ModelLoadError(
                    f"model is not usable (status={status}): "
                    f"{(self.model.error if self.model is not None else 'no model given')}"
                )
            with self.model.lock:
                self._result = _separate_blocking(
                    self.audio_path, self.model, self.config, self._tracker)
        except BaseException as exc:  # noqa: BLE001 - re-raised from result()
            self._error = exc
        finally:
            self._finished = time.time()
            self._done.set()


def _separate_blocking(audio_path: Path, model: LoadedModel,
                       config: SeparationConfig,
                       tracker: progress.Tracker) -> SeparateResult:
    """The separation itself. Runs on Separation's thread, never the caller's.

    The model is already known usable and the model lock already held.
    """
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
        # tracking() redirects the chunk loop's progress bar into `tracker`
        # for this thread only, and is what makes cancel() bite.
        with progress.tracking(tracker):
            produced = model.separator.separate(str(audio_path),
                                                custom_output_names=names)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise errors.OutOfMemory(
            "CUDA ran out of memory. Lower segment_size on the ModelConfig "
            "(try 128) or batch_size on the SeparationConfig, then reload."
        ) from exc
    except errors.Cancelled:
        # Raised by us from inside the chunk loop, so it is not a failure to
        # translate. audio-separator clears the GPU cache and its per-file
        # state on the way out, leaving the model usable for the next job.
        logger.info("separation of %s cancelled after %.1fs",
                    audio_path.name, time.time() - started)
        raise
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
