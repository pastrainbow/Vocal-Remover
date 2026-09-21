"""Fetch, decrypt and tag a Tidal stream.

tidalapi resolves metadata and hands back a StreamManifest, but stops there:
it does not download, decrypt or tag. That is this module's job.

Two manifest shapes exist:
  * BTS  - a single URL
  * MPD  - an MPEG-DASH segment list that must be concatenated in order

Encrypted BTS streams are NOT supported. Tidal can in principle set
encryptionType on a BTS manifest, but it has not been observed for the
qualities this project uses, and carrying an AES-CTR implementation for an
unobserved case meant vendoring Apache-2.0 code and a pycryptodome dependency.
fetch() raises rather than writing a file that would be silently unplayable.
"""
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import mutagen
import requests

from . import errors
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
        part.unlink(missing_ok=True)
        raise errors.DownloadError(
            f"this stream is encrypted (type="
            f"{getattr(manifest, 'encryption_type', '?')}) and decryption support "
            f"was removed as unused. Reinstate an AES-CTR decrypt step in "
            f"_download.fetch() if this starts appearing."
        )

    os.replace(part, dest)
    return dest


def remux_if_needed(path: Path, codecs: str) -> Path:
    """Put a FLAC stream in a .flac container.

    Tidal delivers hi-res FLAC inside an MP4/M4A container, so the file
    arrives as .m4a while actually containing FLAC. That misrepresents the
    content to anything downstream that trusts the extension. `-c:a copy` is a
    container change only - no re-encode, no quality loss, ~0.1s for a track.

    Returns the new path, or the original if no remux was needed or possible.
    """
    if "FLAC" not in str(codecs).upper():
        return path
    if path.suffix.lower() not in (".m4a", ".mp4"):
        return path
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found; leaving FLAC inside %s", path.suffix)
        return path

    target = path.with_suffix(".flac")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path),
             "-map", "0:a", "-c:a", "copy", "-y", str(target)],
            check=True, capture_output=True, timeout=300,
        )
    except Exception as exc:
        logger.warning("remux to .flac failed (%s); keeping %s", exc, path.name)
        target.unlink(missing_ok=True)
        return path

    path.unlink(missing_ok=True)
    logger.debug("remuxed %s -> %s", path.name, target.name)
    return target


def tag(path: Path, info) -> None:
    """Write metadata using mutagen's format-agnostic interface.

    `easy=True` maps a common key set onto whatever the container actually
    uses - Vorbis comments for FLAC, MP4 atoms for m4a - so there is no need
    for per-format branches here. Verified to round-trip title, multi-value
    artist, album and tracknumber identically on both.

    Cover art and ISRC are deliberately not written: neither is expressible
    through the easy interface (EasyMP4 rejects `isrc` outright), and
    supporting them would reintroduce exactly the per-format code this
    replaces. The source file is an intermediate fed to the separation stage,
    and TrackInfo already carries `cover_url` and `isrc` for any UI that wants
    them.

    Best-effort: a tagging failure must never lose the audio.
    """
    path = Path(path)
    try:
        audio = mutagen.File(str(path), easy=True)
        if audio is None:
            logger.debug("mutagen has no handler for %s, leaving untagged",
                         path.suffix)
            return
        audio["title"] = info.title or ""
        audio["artist"] = info.artists or [info.artist or ""]
        if info.album:
            audio["album"] = info.album
        if info.track_number:
            audio["tracknumber"] = str(info.track_number)
        audio.save()
    except Exception:
        logger.warning("tagging failed for %s (audio is intact)", path.name,
                       exc_info=True)
