"""Fetch, decrypt and tag a Tidal stream.

tidalapi resolves metadata and hands back a StreamManifest, but stops there:
it does not download, decrypt or tag. That is this module's job.

Two manifest shapes exist:
  * BTS  - a single URL, occasionally AES-CTR encrypted (encryption_key set)
  * MPD  - an MPEG-DASH segment list that must be concatenated in order
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import requests

from ._vendor.decryption import decrypt_file, decrypt_security_token
from .models import Progress

logger = logging.getLogger("tidal_download")

_CHUNK = 1 << 16


def _safe(name: str, fallback: str = "unknown") -> str:
    """Strip characters Windows refuses in filenames."""
    cleaned = "".join("_" if c in '<>:"/\\|?*' else c for c in str(name or ""))
    cleaned = cleaned.strip().rstrip(".")
    return cleaned[:150] or fallback


def build_filename(track_info, extension: str) -> str:
    ext = extension if extension.startswith(".") else "." + extension
    return f"{_safe(track_info.artist)} - {_safe(track_info.title)}{ext}"


def fetch(manifest, dest: Path,
          progress: Optional[Callable[[Progress], None]] = None) -> Path:
    """Download the stream described by `manifest` to `dest`.

    Returns the final path. Downloads to a .part file first so an interrupted
    run never leaves a truncated file that looks complete.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    urls = list(manifest.urls)
    # A single-URL (BTS) stream reports an exact Content-Length. A segmented
    # MPD stream does not, so total stays 0 and callers show bytes, not a
    # percentage - Progress.fraction already returns 0.0 when total is 0.
    total = 0
    if len(urls) == 1:
        try:
            head = requests.head(urls[0], timeout=30, allow_redirects=True)
            total = int(head.headers.get("content-length") or 0)
        except Exception:
            total = 0

    done = 0
    with open(part, "wb") as out:
        for url in urls:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        try:
                            progress(Progress(downloaded=done, total=total))
                        except Exception:
                            logger.exception("progress callback raised")

    if getattr(manifest, "is_encrypted", False) and manifest.encryption_key:
        logger.debug("decrypting AES-CTR stream")
        key, nonce = decrypt_security_token(manifest.encryption_key)
        tmp = part.with_suffix(part.suffix + ".dec")
        decrypt_file(str(part), str(tmp), key, nonce)
        os.replace(tmp, part)

    os.replace(part, dest)
    return dest


def tag(path: Path, info, cover_bytes: Optional[bytes] = None) -> None:
    """Write metadata. Best-effort: a tagging failure must not lose the audio."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".flac":
            _tag_flac(path, info, cover_bytes)
        elif path.suffix.lower() in (".m4a", ".mp4"):
            _tag_mp4(path, info, cover_bytes)
        else:
            logger.debug("no tagger for %s, leaving untagged", path.suffix)
    except Exception:
        logger.warning("tagging failed for %s (audio is intact)", path.name,
                       exc_info=True)


def _tag_flac(path, info, cover_bytes):
    from mutagen.flac import FLAC, Picture

    audio = FLAC(str(path))
    audio["title"] = info.title or ""
    audio["artist"] = info.artists or [info.artist or ""]
    if info.album:
        audio["album"] = info.album
    if info.track_number:
        audio["tracknumber"] = str(info.track_number)
    if info.isrc:
        audio["isrc"] = info.isrc
    if cover_bytes:
        pic = Picture()
        pic.type, pic.mime, pic.data = 3, "image/jpeg", cover_bytes
        audio.add_picture(pic)
    audio.save()


def _tag_mp4(path, info, cover_bytes):
    from mutagen.mp4 import MP4, MP4Cover

    audio = MP4(str(path))
    audio["\xa9nam"] = info.title or ""
    audio["\xa9ART"] = ", ".join(info.artists) if info.artists else (info.artist or "")
    if info.album:
        audio["\xa9alb"] = info.album
    if info.track_number:
        audio["trkn"] = [(int(info.track_number), 0)]
    if cover_bytes:
        audio["covr"] = [MP4Cover(cover_bytes, imageformat=MP4Cover.FORMAT_JPEG)]
    audio.save()


def fetch_cover(url: Optional[str]) -> Optional[bytes]:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None
