"""Typed exceptions.

audio-separator raises its own exception types plus bare RuntimeError and
torch.cuda.OutOfMemoryError. Everything crossing this package's public
boundary is translated into one of these, so callers never import
audio_separator or torch to handle a failure.
"""


class VocalRemoveError(Exception):
    """Base for every error raised by this package."""


class InvalidConfig(VocalRemoveError):
    """A config value is out of range for its architecture.

    Raised at config construction rather than during separation, because
    several of these produce silently corrupt audio rather than an error -
    e.g. an MDXC-style integer overlap passed to MDX yields a negative step
    and non-finite samples.
    """


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

    Raised separately from SeparationError because the fix is specific:
    lower ModelConfig.segment_size, or reduce batch_size.
    """
