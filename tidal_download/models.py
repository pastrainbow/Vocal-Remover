"""Plain dataclasses for everything crossing the public API boundary.

Callers never touch tidalapi's own model objects, so the library underneath
stays replaceable without breaking the web app.
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Quality(str, Enum):
    """Audio quality, mapped onto tidalapi.Quality in client.py.

    BEST is the default: take the highest quality Tidal will actually serve
    for a given track, walking down MAX -> HIFI -> HIGH -> LOW and stopping at
    the first that works. Availability varies by track and by subscription
    tier - a LOSSLESS-tier account is refused MAX - so a fixed quality fails
    where a lower rung would have succeeded.

    MQA ("Master", Tidal's HI_RES) is deliberately absent. It is a
    lossy-encoded format that separation models gain nothing from, while MAX
    (HI_RES_LOSSLESS) is true hi-res FLAC. This is structural rather than
    guarded: tidalapi 0.8.x has no MQA member to map onto.
    """

    LOW = "LOW"      # AAC ~96k
    HIGH = "HIGH"    # AAC ~320k
    HIFI = "HIFI"    # LOSSLESS - CD-quality FLAC 16/44.1
    MAX = "MAX"      # HI_RES_LOSSLESS - hi-res FLAC, up to 24/192
    BEST = "BEST"    # highest of the above that is actually available


#: Descending preference. MQA/Master is intentionally not a rung.
QUALITY_LADDER = (Quality.MAX, Quality.HIFI, Quality.HIGH, Quality.LOW)


@dataclass(frozen=True)
class AuthState:
    valid: bool
    user_id: Optional[int] = None
    country_code: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class DeviceLogin:
    """A pending device-code login.

    Show verification_url and user_code to the user, then either block on
    TidalClient.await_login() (terminal) or drive TidalClient.poll_login()
    from your own loop (web).
    """

    verification_url: str  # full URL including the code, ready to open
    user_code: str
    expires_in: int  # seconds the code was valid for when issued
    interval: int  # server-requested poll interval
    expires_at: float = 0.0  # absolute epoch deadline; survives serialisation

    @property
    def seconds_left(self) -> int:
        return max(0, int(self.expires_at - time.time()))

    @property
    def expired(self) -> bool:
        return self.seconds_left <= 0


@dataclass(frozen=True)
class TrackInfo:
    id: int
    title: str
    artist: str
    artists: List[str] = field(default_factory=list)
    album: Optional[str] = None
    album_id: Optional[int] = None
    duration: int = 0
    track_number: Optional[int] = None
    explicit: bool = False
    isrc: Optional[str] = None
    cover_url: Optional[str] = None

    @property
    def display(self) -> str:
        return f"{self.artist} - {self.title}"


@dataclass(frozen=True)
class DownloadResult:
    path: str
    track: TrackInfo
    quality: str  # what Tidal actually served, which may differ from requested
    codec: str
    size_bytes: int


@dataclass(frozen=True)
class Progress:
    """Passed to the download progress callback."""

    downloaded: int
    total: int

    @property
    def fraction(self) -> float:
        return (self.downloaded / self.total) if self.total else 0.0
