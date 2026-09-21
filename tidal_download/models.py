"""Plain dataclasses for everything crossing the public API boundary.

Callers never touch the vendored aigpy model objects, so the vendored tree
stays replaceable without breaking the web app.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Quality(str, Enum):
    """Audio quality, mapped onto the vendored AudioQuality enum.

    BEST is the default: take the highest quality Tidal will actually serve
    for a given track, walking down MAX -> HIFI -> HIGH -> LOW and stopping at
    the first that works. Availability varies per track and per API key, so a
    fixed quality fails on tracks that a lower rung would have served.

    MQA ("Master", Tidal's HI_RES) is deliberately absent and is refused in the
    vendored getStreamUrl. It is a lossy-encoded format that separation models
    gain nothing from; MAX (HI_RES_LOSSLESS) is true hi-res FLAC.
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
    expires_in: Optional[int] = None  # seconds; negative means expired
    api_key_index: Optional[int] = None
    api_key_platform: Optional[str] = None
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
        import time

        return max(0, int(self.expires_at - time.time()))

    @property
    def expired(self) -> bool:
        return self.seconds_left <= 0


@dataclass(frozen=True)
class ApiKeyInfo:
    index: int
    platform: str
    formats: str
    valid: bool
    current: bool


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
