"""Does python-tidal get playback where the vendored keys cannot?

DEV SCRIPT. Proves (or disproves) the premise behind swapping the auth and
streaming layer, before any of tidal_download is rewritten around it.

The vendored yaronzz client IDs authenticate fine but every playbackinfo
request returns 401 / subStatus 4005 - Tidal revoked playback for those leaked
2023-era credentials. tidalapi ships its own, actively maintained ones. If this
script reaches a stream URL, the swap is worth doing; if it returns 4005 too,
the problem is the account or region and no library change will help.

    venv\\Scripts\\python.exe -m smoke_test.tidalapi_probe <track-url-or-id>
"""
import sys

import tidalapi

DEFAULT_TRACK = 242542596

LADDER = [
    ("MAX", tidalapi.Quality.hi_res_lossless),
    ("HIFI", tidalapi.Quality.high_lossless),
    ("HIGH", tidalapi.Quality.low_320k),
    ("LOW", tidalapi.Quality.low_96k),
]


def track_id_from(arg):
    if arg is None:
        return DEFAULT_TRACK
    digits = "".join(c for c in str(arg).split("?")[0] if c.isdigit() or c == "/")
    for part in reversed(digits.split("/")):
        if part.isdigit():
            return int(part)
    raise SystemExit(f"could not find a track id in {arg!r}")


def main():
    track_id = track_id_from(sys.argv[1] if len(sys.argv) > 1 else None)

    session = tidalapi.Session()
    print("=" * 66)
    print("TIDALAPI PLAYBACK PROBE")
    print("=" * 66)
    print(f"  client id     : {session.config.client_id}")
    print(f"  track         : {track_id}")
    print()

    login, future = session.login_oauth()
    print(f"  1. open   : https://{login.verification_uri_complete}")
    print(f"  2. code   : {login.user_code}")
    print(f"  3. approve; this continues automatically "
          f"(expires in {int(login.expires_in)}s)")
    print()
    future.result()  # blocks until approved or the code expires

    if not session.check_login():
        print("  LOGIN FAILED")
        return 1
    print(f"  logged in     : user {session.user.id} ({session.country_code})")
    print()

    track = session.track(track_id)
    print(f"  title         : {track.name}")
    print(f"  artist        : {track.artist.name if track.artist else '?'}")
    print(f"  stream_ready  : {getattr(track, 'stream_ready', '?')}")
    print(f"  allow_stream  : {getattr(track, 'allow_streaming', '?')}")
    print()

    best = None
    for label, quality in LADDER:
        session.audio_quality = quality
        try:
            stream = track.get_stream()
            url_ok = bool(track.get_url())
            print(f"  {label:<5} {quality.value:<18} OK  "
                  f"codec={stream.audio_quality} {stream.sample_rate}Hz/"
                  f"{stream.bit_depth}bit url={'yes' if url_ok else 'no'}")
            best = best or label
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:90]
            print(f"  {label:<5} {quality.value:<18} FAIL {type(exc).__name__}: {msg}")

    print("=" * 66)
    if best:
        print(f"PLAYBACK WORKS - best available: {best}")
        print("The vendored keys are the problem; the tidalapi swap is worth doing.")
        return 0
    print("PLAYBACK STILL BLOCKED")
    print("Not a library problem - the account, region or track is the blocker.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
