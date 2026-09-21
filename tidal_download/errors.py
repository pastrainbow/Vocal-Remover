"""Typed exceptions.

The vendored code signals failure by returning (ok, err_string) tuples and by
raising bare Exception. Everything crossing this package's public boundary is
translated into one of these instead.
"""


class TidalError(Exception):
    """Base for every error raised by this package."""


class AuthError(TidalError):
    """No usable token: never logged in, or refresh failed."""


class ApiKeyError(TidalError):
    """The bundled API key was rejected.

    These keys are hardcoded upstream and rot whenever Tidal rotates them.
    Recovery is to try a different index; see TidalClient.list_api_keys().
    """


class LoginTimeout(TidalError):
    """The device-code flow was not approved before the code expired."""


class UnsupportedUrl(TidalError):
    """URL parsed, but is not a media type this package handles."""


class NotFound(TidalError):
    """The URL parsed and is supported, but Tidal returned nothing."""


class DownloadError(TidalError):
    """Stream fetch, decrypt, or tagging failed."""
