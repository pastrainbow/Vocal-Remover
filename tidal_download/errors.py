"""Typed exceptions.

tidalapi surfaces failures as requests.HTTPError, bare Exception and library
specific types. Everything crossing this package's public boundary is
translated into one of these instead, so callers never import tidalapi to
handle an error.
"""


class TidalError(Exception):
    """Base for every error raised by this package."""


class AuthError(TidalError):
    """No usable token: never logged in, or refresh failed."""


class LoginTimeout(TidalError):
    """The device-code flow was not approved before the code expired."""


class UnsupportedUrl(TidalError):
    """URL parsed, but is not a media type this package handles."""


class NotFound(TidalError):
    """The URL parsed and is supported, but Tidal returned nothing."""


class DownloadError(TidalError):
    """No quality was available, or the stream could not be fetched.

    Also raised for an encrypted stream: decryption support was removed as
    unused, so fetch() refuses rather than writing an unplayable file.
    """
