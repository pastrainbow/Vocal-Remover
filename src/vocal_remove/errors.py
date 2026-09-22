"""Typed exceptions.

audio-separator raises its own exception types plus bare RuntimeError and
torch.cuda.OutOfMemoryError. Everything crossing this package's public
boundary is translated into one of these, so callers never import
audio_separator or torch to handle a failure.
"""


class VocalRemoveError(Exception):
    """Base for every error raised by this package."""


class ModelLoadError(VocalRemoveError):
    """The model could not be downloaded, found, or loaded onto the device."""


class AudioNotFound(VocalRemoveError):
    """The input audio file does not exist or could not be read."""


class Cancelled(VocalRemoveError):
    """The separation was stopped by Separation.cancel().

    Deliberately NOT a SeparationError: nothing failed, the caller asked for
    this, and a caller that cancels usually wants to tell the two apart.
    """


class SeparationError(VocalRemoveError):
    """Separation ran but did not produce the expected stems."""


class OutOfMemory(SeparationError):
    """The GPU ran out of memory.

    Raised separately from SeparationError because it is a capacity problem
    rather than a broken job: the same work would succeed on a card with more
    free VRAM. There is no inference parameter to turn down - see config.py -
    so the levers are freeing VRAM on the device or keeping fewer models
    resident.
    """
