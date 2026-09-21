"""tidal_download - a small library API over an adapted tidal-media-downloader.

    from tidal_download import TidalClient, Quality

    client = TidalClient()
    if not client.auth_status().valid:
        dev = client.begin_login()
        print("open", dev.verification_url, "code", dev.user_code)
        client.await_login(dev)

    info = client.resolve(url)
    res = client.download(url, "data/staging/job123", quality=Quality.HIFI)
    print(res.path)

The vendored upstream tree lives in _vendor/ and is Apache-2.0; see NOTICE for
the modifications made to it.
"""
from .client import TidalClient
from .errors import (
    AuthError,
    DownloadError,
    LoginTimeout,
    NotFound,
    TidalError,
    UnsupportedUrl,
)
from .models import (
    AuthState,
    DeviceLogin,
    DownloadResult,
    Progress,
    QUALITY_LADDER,
    Quality,
    TrackInfo,
)

__all__ = [
    "TidalClient",
    "Quality",
    "QUALITY_LADDER",
    "AuthState",
    "DeviceLogin",
    "TrackInfo",
    "DownloadResult",
    "Progress",
    "TidalError",
    "AuthError",
    "LoginTimeout",
    "UnsupportedUrl",
    "NotFound",
    "DownloadError",
]
