"""A thin adapter over python-tidal (tidalapi).

Its only job is to keep tidalapi's types and failure modes out of the rest of
the app: callers get plain dataclasses and typed errors, and never import
tidalapi to handle a result or an exception.

That boundary has already earned its keep once. This package previously wrapped
a vendored downloader whose credentials Tidal revoked for playback; replacing
the entire implementation with tidalapi changed nothing outside this file.

What tidalapi does NOT provide, and therefore lives here:
  * URL -> track id parsing
  * picking the best quality the account can actually stream
  * downloading, remuxing and tagging (in _download.py)
"""
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Union

import tidalapi

from . import errors
from ._download import build_filename, fetch, remux_if_needed, tag
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

#: tidalapi 0.8.x has no MQA member, so "never download Master" is structural.
_QUALITY_MAP = {
    Quality.LOW: tidalapi.Quality.low_96k,
    Quality.HIGH: tidalapi.Quality.low_320k,
    Quality.HIFI: tidalapi.Quality.high_lossless,
    Quality.MAX: tidalapi.Quality.hi_res_lossless,
}


def _track_info(track) -> TrackInfo:
    """tidalapi Track -> our TrackInfo. The whole point of the adapter."""
    artists = [a.name for a in (track.artists or [])] if track.artists else []
    album = track.album
    try:
        cover = album.image(640) if album else None
    except Exception:
        cover = None
    return TrackInfo(
        id=track.id,
        title=track.name,
        artist=track.artist.name if track.artist else (artists[0] if artists else ""),
        artists=artists,
        album=getattr(album, "name", None),
        album_id=getattr(album, "id", None),
        duration=int(track.duration or 0),
        track_number=track.track_num,
        explicit=bool(track.explicit),
        isrc=getattr(track, "isrc", None),
        cover_url=cover,
    )


class TidalClient:
    """A Tidal session scoped to one config directory.

    >>> client = TidalClient()
    >>> if not client.auth_status().valid:
    ...     dev = client.begin_login()
    ...     print(dev.verification_url, dev.user_code)
    ...     while client.poll_login(dev) is None:
    ...         time.sleep(dev.interval)
    >>> client.download("https://tidal.com/track/1234", "out/").path
    """

    def __init__(self, config_dir: Union[str, Path, None] = None):
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[1] / "state" / "tidal"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.session_file = self.config_dir / "session.json"

        self.session = tidalapi.Session()
        try:
            self.session.load_session_from_file(self.session_file)
        except Exception:
            # A stale or foreign session file must not stop the client being
            # constructed - auth_status() will simply report "not logged in".
            logger.warning("ignoring unreadable session file %s",
                           self.session_file, exc_info=True)

    # ---------------------------------------------------------------- auth

    def auth_status(self) -> AuthState:
        """Verify the stored session, refreshing it if it has expired."""
        if not self.session.access_token:
            return AuthState(False, detail="no session stored")
        try:
            if self.session.check_login():
                return self._ok("session valid")
            if self.session.refresh_token and \
                    self.session.token_refresh(self.session.refresh_token):
                self.session.save_session_to_file(self.session_file)
                return self._ok("session refreshed")
        except Exception as exc:
            return AuthState(False, detail=f"verify failed: {exc}")
        return AuthState(False, detail="session rejected and refresh failed")

    def _ok(self, detail: str) -> AuthState:
        user = getattr(self.session, "user", None)
        return AuthState(True, getattr(user, "id", None),
                         self.session.country_code, detail=detail)

    def begin_login(self) -> DeviceLogin:
        """Start the device-code flow; show the result to the user."""
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
        """Check ONCE whether the code was approved; None while still pending.

        Deliberately non-blocking: tidalapi's own helpers wait for the full
        code lifetime, which would tie up a web request handler for minutes.
        Callers loop over this at `device.interval`.
        """
        if device.expired:
            raise errors.LoginTimeout("device code expired before it was approved")
        pending = getattr(self, "_pending", None)
        if pending is None:
            raise errors.AuthError("begin_login() must be called before poll_login()")
        try:
            approved = self.session.process_link_login(pending, until_expiry=False)
        except TimeoutError:
            return None
        except Exception as exc:
            logger.warning("auth poll failed, retrying: %s: %s",
                           type(exc).__name__, exc)
            return None
        if not approved:
            return None
        self.session.save_session_to_file(self.session_file)
        return self._ok("logged in")

    def logout(self) -> None:
        self.session_file.unlink(missing_ok=True)
        self.session = tidalapi.Session()

    # ------------------------------------------------------------- content

    def resolve(self, url: str) -> TrackInfo:
        """Metadata for a track URL, without downloading it.

        Lets a caller validate a URL and show what it points at before
        committing to a job.
        """
        return _track_info(self._track(url))

    def download(
        self,
        url: str,
        out_dir: Union[str, Path],
        quality: Quality = Quality.BEST,
        progress: Optional[Callable[[Progress], None]] = None,
    ) -> DownloadResult:
        """Download one track into out_dir and return where it landed.

        out_dir should be empty and job-specific; the file is written flat
        inside it, with no artist/album folder nesting.
        """
        track = self._track(url)
        info = _track_info(track)
        effective, stream = self._best_stream(track, Quality(quality))

        manifest = stream.get_stream_manifest()
        dest = Path(out_dir) / build_filename(info, manifest.file_extension)
        dest.parent.mkdir(parents=True, exist_ok=True)

        path = fetch(manifest, dest, progress=progress)
        # Tidal ships hi-res FLAC inside an MP4 container; give it a .flac one
        # so the extension matches the codec for everything downstream.
        path = remux_if_needed(path, manifest.codecs)
        tag(path, info)

        return DownloadResult(
            path=str(path),
            track=info,
            quality=getattr(stream, "audio_quality", effective.value),
            codec=str(manifest.codecs),
            size_bytes=path.stat().st_size,
        )

    # ------------------------------------------------------------ internal

    #: Resource words that appear in Tidal URL paths. Anything other than
    #: "track" is a different kind of thing and must not be downloaded as one.
    _NON_TRACK_RESOURCES = ("album", "playlist", "mix", "artist", "video")

    @classmethod
    def _track_id(cls, url: str) -> int:
        """Accept a track URL in any Tidal shape, or a bare id.

        The resource word is checked, not just the trailing number. Album and
        playlist URLs also end in a numeric id, so matching on digits alone
        silently downloads whatever TRACK happens to share that id - a real
        bug this guard exists to prevent.
        """
        parts = [p for p in str(url).split("?")[0].strip("/").split("/") if p]
        lowered = [p.lower() for p in parts]

        for resource in cls._NON_TRACK_RESOURCES:
            if resource in lowered:
                article = "an" if resource[0] in "aeiou" else "a"
                raise errors.UnsupportedUrl(
                    f"{url!r} is {article} {resource} URL; only single tracks "
                    f"are supported"
                )

        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        raise errors.UnsupportedUrl(
            f"no track id found in {url!r}; only single tracks are supported"
        )

    def _track(self, url: str):
        state = self.auth_status()
        if not state.valid:
            raise errors.AuthError(f"not authenticated: {state.detail}")
        track_id = self._track_id(url)
        try:
            return self.session.track(track_id)
        except Exception as exc:
            raise errors.NotFound(f"could not resolve {url}: {exc}") from exc

    def _best_stream(self, track, quality: Quality):
        """(Quality, stream) for the highest rung this account can stream.

        Availability depends on subscription tier as well as the track - a
        LOSSLESS-tier account is refused MAX - so a fixed quality fails where
        a lower rung would have succeeded.
        """
        rungs = list(QUALITY_LADDER) if quality is Quality.BEST else [quality]
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
            "no audio quality available for this track - " + "; ".join(failures)
        )
