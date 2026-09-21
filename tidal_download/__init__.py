"""tidal_download - a thin adapter over python-tidal (tidalapi).

    from tidal_download import TidalClient

    client = TidalClient()
    if not client.auth_status().valid:
        dev = client.begin_login()
        print("open", dev.verification_url, "code", dev.user_code)
        while client.poll_login(dev) is None:     # non-blocking, caller paces
            time.sleep(dev.interval)

Somewhere that cannot sit in that loop - a web request handler, a UI thread -
uses client.login instead, which runs it on a thread and answers instantly:

    status = client.login.begin()                # returns with a code
    status = client.login.status()               # poll this as often as you like
    status.stage                                 # pending -> ok / expired / error

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
from .login import LoginFlow, LoginStage, LoginStatus
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
    "LoginFlow",
    "LoginStage",
    "LoginStatus",
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
