"""Chunk-level progress and cancellation for a running separation.

audio-separator offers neither: separate() blocks and returns only when it is
finished. Both are recoverable from the same place. Every architecture
processes the track as a sequence of chunks, wraps that loop in a tqdm bar,
and binds tqdm with `from tqdm import tqdm` at module import. Replacing that
module attribute with a proxy puts this package inside the loop, where it can
count chunks as they finish and raise at a chunk boundary to stop the run.

This is a PRIVATE-API dependency, pinned to audio-separator 0.47.0 and
verified by reading it:

  mdxc_separator.demix()  one chunk loop per call (the roformer and mdx23c
                          branches are exclusive), one demix() per separate()
  mdx_separator.demix()   one chunk loop per call, but separate() calls
                          demix() TWICE - once for the primary stem, then
                          again in match-mix mode to derive the secondary
  vr_separator            several loops of very uneven size: a 4-item band
                          loop, a trivial slicing loop, then the real
                          inference loop - so a chunk count says little

Hence _PASS_WEIGHTS below, which models only the two architectures whose loops
were read and measured. Anything else runs tracked but unweighted: it can
still be cancelled, and reports indeterminate progress rather than a number
that would be wrong.

If a future audio-separator moves or renames these loops, progress stalls and
cancellation stops working; separation itself does not. The proxy delegates to
the real tqdm and is transparent on any thread that is not being tracked.
"""
import sys
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

from . import errors

#: Modules whose `tqdm` attribute is swapped. Demucs is absent on purpose: it
#: reaches tqdm as `tqdm.tqdm` from uvr_lib_v5/demucs/utils.py, so there is no
#: module attribute to swap without patching tqdm for the whole process.
#: Demucs therefore runs untracked - no progress, no cancellation.
_ARCH_MODULES = (
    "audio_separator.separator.architectures.mdxc_separator",
    "audio_separator.separator.architectures.mdx_separator",
    "audio_separator.separator.architectures.vr_separator",
)

#: Relative cost of each chunk loop one separate() runs, keyed by the
#: architecture class audio-separator instantiates.
#:
#: Weighted by TIME, not chunk count: MDX's second pass re-demixes the whole
#: track in match-mix mode, where run_model() skips inference entirely and
#: only round-trips the STFT, so its chunks are far cheaper than the first
#: pass's. Its 0.25 is measured - 11.9s then 2.9s on the 229.5s test track
#: with UVR-MDX-NET-Voc_FT on an RTX 4060 - rather than derived, because the
#: chunk counts (0.77x) say the opposite of what the clock does.
#:
#: Job settings bend this: a higher overlap lengthens the first pass only, and
#: enable_denoise runs the model twice per chunk there. Both make the curve
#: uneven; neither can make it run backwards.
_PASS_WEIGHTS: Dict[str, Tuple[float, ...]] = {
    "MDXCSeparator": (1.0,),
    "MDXSeparator": (1.0, 0.25),
}

_REAL_TQDM = None
_install_lock = threading.Lock()
#: Thread id -> the Tracker collecting that thread's bars. Separations run one
#: per thread, so this is what attributes a chunk loop to the right job.
_active: Dict[int, "Tracker"] = {}


class Tracker:
    """Counts chunks for one separation, and carries its cancel flag.

    Progress is reported as a fraction of the chunk work only; what happens
    before the first chunk (decoding the input) and after the last (writing
    the stems) is the caller's to account for.
    """

    def __init__(self, weights: Optional[Tuple[float, ...]] = None):
        self._weights = weights
        self._weight_total = float(sum(weights)) if weights else 0.0
        self._guard = threading.Lock()
        self._pass = 0        # how many chunk loops have started
        self._base = 0.0      # where the running loop starts in [0, 1]
        self._span = 0.0      # how much of [0, 1] it covers
        self._fraction = 0.0
        self._cancelled = False

    @property
    def indeterminate(self) -> bool:
        """True when this architecture's loops are not modelled."""
        return self._weights is None

    @property
    def fraction(self) -> float:
        """Chunk work completed, 0.0 to 1.0. Never decreases."""
        with self._guard:
            return self._fraction

    @property
    def cancelled(self) -> bool:
        with self._guard:
            return self._cancelled

    def cancel(self) -> None:
        """Stop the separation at the next chunk boundary."""
        with self._guard:
            self._cancelled = True

    # ---------------------------------------------------------- interception

    def _consume(self, bar):
        """Iterate one tqdm bar, counting it and honouring cancellation."""
        total = float(getattr(bar, "total", 0) or 0)
        self._open_pass()
        try:
            for done, item in enumerate(bar, start=1):
                # Before handing the chunk over, not after: this is the last
                # point where stopping costs nothing.
                if self.cancelled:
                    raise errors.Cancelled("separation cancelled")
                yield item
                if total:
                    self._advance(done / total)
            self._close_pass()
        finally:
            # The bar is only closed by running out, so close it by hand for
            # the cancelled and failed paths.
            bar.close()

    def _open_pass(self) -> None:
        with self._guard:
            index = self._pass
            self._pass += 1
            if self._weights is not None and index < len(self._weights):
                self._base = sum(self._weights[:index]) / self._weight_total
                self._span = self._weights[index] / self._weight_total
            else:
                # More loops than _PASS_WEIGHTS knows about, so there is no
                # range left to give this one: progress holds where the
                # modelled passes left it until the run ends. That means the
                # weights need re-measuring against a new audio-separator.
                self._base = self._fraction
                self._span = 0.0

    def _close_pass(self) -> None:
        with self._guard:
            self._fraction = max(self._fraction, self._base + self._span)

    def _advance(self, within_pass: float) -> None:
        with self._guard:
            self._fraction = max(self._fraction,
                                 self._base + self._span * within_pass)


def tracker_for(model_instance) -> Tracker:
    """A Tracker weighted for whichever architecture this model is."""
    arch = type(model_instance).__name__ if model_instance is not None else ""
    return Tracker(_PASS_WEIGHTS.get(arch))


@contextmanager
def tracking(tracker: Tracker):
    """Route the calling thread's chunk loops into `tracker`."""
    _install()
    ident = threading.get_ident()
    _active[ident] = tracker
    try:
        yield tracker
    finally:
        _active.pop(ident, None)


def _tqdm_proxy(iterable=None, *args, **kwargs):
    tracker = _active.get(threading.get_ident())
    bar = _REAL_TQDM(iterable, *args, **kwargs)
    if tracker is None or iterable is None:
        # Not a tracked separation, or a bar the library drives by hand
        # (total= with .update()) rather than by iterating. Leave it alone.
        return bar
    return tracker._consume(bar)


_tqdm_proxy._vocal_remove_proxy = True


def _install() -> None:
    """Swap tqdm in every architecture module that is already imported.

    Deliberately not at import time, and deliberately not importing anything:
    importing an architecture we are not using would drag in its runtime
    (onnxruntime, for one) for nothing. Every separation re-checks, so an
    architecture whose model loads later is still picked up.
    """
    global _REAL_TQDM
    with _install_lock:
        for name in _ARCH_MODULES:
            module = sys.modules.get(name)
            if module is None:
                continue
            current = getattr(module, "tqdm", None)
            if current is None or getattr(current, "_vocal_remove_proxy", False):
                continue
            # All three bind the same tqdm class, so one saved reference is
            # enough for the proxy to delegate to.
            if _REAL_TQDM is None:
                _REAL_TQDM = current
            module.tqdm = _tqdm_proxy
