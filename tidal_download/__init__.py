"""tidal_download - a thin adapter over python-tidal (tidalapi).

    from tidal_download import TidalClient

    client = TidalClient()
    if not client.auth_status().valid:
        dev = client.begin_login()
        print("open", dev.verification_url, "code", dev.user_code)
        while client.poll_login(dev) is None:     # non-blocking, caller paces
            time.sleep(dev.interval)

    info = client.resolve(url)                   # metadata, no download
    res = client.download(url, "data/staging/job123")
    print(res.path)                              # quality defaults to BEST

Auth, metadata and stream resolution come from tidalapi; downloading, remuxing
and tagging are in _download.py. No vendored code.
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
