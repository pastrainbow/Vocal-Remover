"""TidalClient - the public interface.

Built on python-tidal (tidalapi). The previous implementation wrapped the
vendored yaronzz client, whose hardcoded credentials Tidal has revoked for
playback: they still authenticate, but every playbackinfo request returns
401 / subStatus 4005 regardless of quality, paywall endpoint or subscription
state. tidalapi ships maintained credentials and was verified against the same
account and track that the vendored keys could not play.

Only _vendor/decryption.py survives from that tree - tidalapi resolves streams
but does not decrypt them.

Threading note
--------------
Each TidalClient owns its own tidalapi.Session, so instances no longer share
process-global state and downloads no longer have to be serialised. The token
file is still shared, so writes to it are guarded by _TOKEN_LOCK.
"""
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import tidalapi

from . import errors
from ._download import build_filename, fetch, fetch_cover, tag
from .models import (
    AuthState,
    DeviceLogin,
    DownloadResult,
    Progress,
    QUALITY_LADDER,
    Quality,
    TrackInfo,
)

logger = logging.getLogger("tidal_download")

_TOKEN_LOCK = threading.RLock()

#: Our Quality -> tidalapi.Quality. tidalapi 0.8.x has no MQA member at all,
#: so "never download Master" is structural here rather than a guard we apply.
_QUALITY_MAP = {
    Quality.LOW: tidalapi.Quality.low_96k,
    Quality.HIGH: tidalapi.Quality.low_320k,
    Quality.HIFI: tidalapi.Quality.high_lossless,
    Quality.MAX: tidalapi.Quality.hi_res_lossless,
}


def _track_info(track) -> TrackInfo:
    artists = [a.name for a in (track.artists or [])] if getattr(track, "artists", None) else []
    artist = track.artist.name if getattr(track, "artist", None) else (artists[0] if artists else "")
    album = getattr(track, "album", None)
    cover = None
    if album is not None:
        try:
            cover = album.image(640)
        except Exception:
            cover = None
    return TrackInfo(
        id=track.id,
        title=track.name,
        artist=artist,
        artists=artists,
        album=getattr(album, "name", None),
        album_id=getattr(album, "id", None),
        duration=int(getattr(track, "duration", 0) or 0),
        track_number=getattr(track, "track_num", None),
        explicit=bool(getattr(track, "explicit", False)),
        isrc=getattr(track, "isrc", None),
        cover_url=cover,
    )


class TidalClient:
    """A Tidal session scoped to one config directory.

    >>> client = TidalClient()
    >>> if not client.auth_status().valid:
    ...     dev = client.begin_login()
    ...     print(dev.verification_url, dev.user_code)
    ...     client.await_login(dev)
    >>> client.download("https://tidal.com/track/1234", "out/").path
    """

    def __init__(self, config_dir: Union[str, Path, None] = None):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[1] / "state" / "tidal"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = self.config_dir / "session.json"

        self.session = tidalapi.Session()
        self._restore()

    # ---------------------------------------------------------------- token

    def _restore(self) -> bool:
        if not self.token_path.is_file():
            return False
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
            expiry = data.get("expiry_time")
            return self.session.load_oauth_session(
                token_type=data["token_type"],
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token"),
                expiry_time=datetime.fromisoformat(expiry) if expiry else None,
            )
        except Exception:
            logger.warning("could not restore session from %s", self.token_path,
                           exc_info=True)
            return False

    def _persist(self) -> None:
        expiry = getattr(self.session, "expiry_time", None)
        payload = {
            "token_type": self.session.token_type,
            "access_token": self.session.access_token,
            "refresh_token": self.session.refresh_token,
            "expiry_time": expiry.isoformat() if isinstance(expiry, datetime) else None,
        }
        with _TOKEN_LOCK:
            self.token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("session saved to %s", self.token_path)

    # ---------------------------------------------------------------- auth

    def auth_status(self) -> AuthState:
        """Verify the stored session, refreshing it if it has expired."""
        if not self.session.access_token:
            return AuthState(False, detail="no token stored")
        try:
            if self.session.check_login():
                return self._state("session valid")
            if self.session.refresh_token and \
                    self.session.token_refresh(self.session.refresh_token):
                self._persist()
                return self._state("session refreshed")
        except Exception as exc:
            return AuthState(False, detail=f"verify failed: {exc}")
        return AuthState(False, detail="session rejected and refresh failed")

    def _state(self, detail: str) -> AuthState:
        expiry = getattr(self.session, "expiry_time", None)
        secs = None
        if isinstance(expiry, datetime):
            now = datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.now()
            secs = int((expiry - now).total_seconds())
        user_id = getattr(getattr(self.session, "user", None), "id", None)
        return AuthState(True, user_id, self.session.country_code, secs,
                         detail=detail)

    def begin_login(self) -> DeviceLogin:
        """Start the device-code flow.

        Show the result to the user, then either block on await_login() or
        drive poll_login() from your own loop.
        """
        try:
            login = self.session.get_link_login()
        except Exception as exc:
            raise errors.AuthError(f"could not start device login: {exc}") from exc
        self._pending = login
        return DeviceLogin(
            verification_url="https://" + login.verification_uri_complete,
            user_code=login.user_code,
            expires_in=int(login.expires_in),
            interval=int(login.interval),
            expires_at=time.time() + float(login.expires_in),
        )

    def poll_login(self, device: DeviceLogin) -> Optional[AuthState]:
        """Check ONCE whether the device code has been approved.

        Returns an AuthState once approved (session persisted), or None while
        still pending. Raises LoginTimeout once the code has expired.

        This is the primitive a web app wants: no request handler blocks for
        minutes. await_login() is just this in a loop.
        """
        if device.expired:
            raise errors.LoginTimeout("device code expired before it was approved")
        pending = getattr(self, "_pending", None)
        if pending is None:
            raise errors.AuthError("begin_login() must be called before poll_login()")
        try:
            # until_expiry=False makes this a single poll rather than a
            # blocking wait, which is what tidalapi does by default.
            ok = self.session.process_link_login(pending, until_expiry=False)
        except TimeoutError:
            return None  # not yet approved
        except Exception as exc:
            logger.warning("auth poll failed, retrying: %s: %s",
                           type(exc).__name__, exc)
            return None
        if not ok:
            return None
        self._persist()
        return self._state("logged in")

    def await_login(self, device: DeviceLogin,
                    on_tick: Optional[Callable[[int], None]] = None) -> AuthState:
        """Block until approved, then persist. Terminal convenience wrapper."""
        while not device.expired:
            state = self.poll_login(device)
            if state is not None:
                return state
            if on_tick:
                on_tick(device.seconds_left)
            time.sleep(device.interval + 1)
        raise errors.LoginTimeout("device code expired before it was approved")

    def login(self, on_prompt: Callable[[DeviceLogin], None],
              on_tick: Optional[Callable[[int], None]] = None) -> AuthState:
        device = self.begin_login()
        on_prompt(device)
        return self.await_login(device, on_tick)

    def logout(self) -> None:
        with _TOKEN_LOCK:
            if self.token_path.is_file():
                self.token_path.unlink()
        self.session = tidalapi.Session()

    # ------------------------------------------------------------- content

    @staticmethod
    def _track_id(url: str) -> int:
        """Accept a track URL in any Tidal shape, or a bare id."""
        text = str(url).split("?")[0].rstrip("/")
        parts = [p for p in text.split("/") if p]
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        raise errors.UnsupportedUrl(
            f"no track id found in {url!r}; only single tracks are supported"
        )

    def _require_auth(self):
        state = self.auth_status()
        if not state.valid:
            raise errors.AuthError(f"not authenticated: {state.detail}")

    def resolve(self, url: str) -> TrackInfo:
        """Resolve a Tidal track URL to its metadata. No download."""
        self._require_auth()
        try:
            track = self.session.track(self._track_id(url))
        except errors.TidalError:
            raise
        except Exception as exc:
            raise errors.NotFound(f"could not resolve {url}: {exc}") from exc
        return _track_info(track)

    def resolve_quality(self, track, ceiling: Quality = Quality.MAX):
        """Return (Quality, stream) for the best quality Tidal will actually serve.

        Walks QUALITY_LADDER downward from `ceiling`. Availability depends on
        the subscription tier as well as the track - a LOSSLESS-tier account
        gets 401 on MAX - so a fixed quality fails where a lower rung succeeds.
        """
        rungs = list(QUALITY_LADDER)
        if ceiling in rungs:
            rungs = rungs[rungs.index(ceiling):]
        failures = []
        for rung in rungs:
            self.session.audio_quality = _QUALITY_MAP[rung]
            try:
                stream = track.get_stream()
            except Exception as exc:
                failures.append(f"{rung.value}: {type(exc).__name__}")
                continue
            if rung is not rungs[0]:
                logger.info("%s unavailable, using %s", rungs[0].value, rung.value)
            return rung, stream
        raise errors.DownloadError(
            "no audio quality is available for this track - " + "; ".join(failures)
        )

    def download(
        self,
        url: str,
        out_dir: Union[str, Path],
        quality: Quality = Quality.BEST,
        progress: Optional[Callable[[Progress], None]] = None,
    ) -> DownloadResult:
        """Download one track into out_dir and return where it landed.

        out_dir should be empty and job-specific. The file is written flat
        inside it, with no artist/album folder nesting.
        """
        self._require_auth()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            track = self.session.track(self._track_id(url))
        except errors.TidalError:
            raise
        except Exception as exc:
            raise errors.NotFound(f"could not resolve {url}: {exc}") from exc

        info = _track_info(track)
        requested = Quality(quality)
        if requested is Quality.BEST:
            effective, stream = self.resolve_quality(track)
        else:
            effective = requested
            self.session.audio_quality = _QUALITY_MAP[effective]
            try:
                stream = track.get_stream()
            except Exception as exc:
                raise errors.DownloadError(
                    f"{effective.value} is not available for this track: {exc}"
                ) from exc

        manifest = stream.get_stream_manifest()
        dest = out_dir / build_filename(info, manifest.file_extension)
        path = fetch(manifest, dest, progress=progress)
        tag(path, info, fetch_cover(info.cover_url))

        return DownloadResult(
            path=str(path),
            track=info,
            quality=getattr(stream, "audio_quality", effective.value),
            codec=str(getattr(manifest, "codecs", "")),
            size_bytes=path.stat().st_size,
        )
