"""Manual smoke-test CLI for tidal_download.

DEV SCRIPT - not imported by the web app. It is a thin shell over the public
API in tidal_download.client and holds no logic of its own, so anything it
can do, the web app can do too by calling the same methods.

    python -m smoke_test.tidal_cli status
    python -m smoke_test.tidal_cli login
    python -m smoke_test.tidal_cli resolve <url>
    python -m smoke_test.tidal_cli get <url> -o data/staging/test
    python -m smoke_test.tidal_cli logout

Run with:
    venv/Scripts/python.exe -m smoke_test.tidal_cli status
"""
import argparse
import logging
import sys
import time

from tidal_download import TidalClient, Quality, errors


def _banner(msg):
    print("\n" + "=" * 64)
    print(msg)
    print("=" * 64)


def _print_auth(state):
    print("  authenticated : %s" % ("yes" if state.valid else "NO"))
    print("  detail        : %s" % state.detail)
    if state.valid:
        print("  user          : %s (%s)" % (state.user_id, state.country_code))


def cmd_status(client, _args):
    _banner("TIDAL AUTH STATUS")
    state = client.auth_status()
    print("  config dir    : %s" % client.config_dir)
    _print_auth(state)
    return 0 if state.valid else 1


def cmd_login(client, args):
    state = client.auth_status()
    if state.valid and not args.force:
        _banner("ALREADY LOGGED IN")
        _print_auth(state)
        print("\n  use --force to log in again")
        return 0

    _banner("TIDAL DEVICE LOGIN")

    def on_prompt(dev):
        print("\n  1. open   : %s" % dev.verification_url)
        print("  2. code   : %s" % dev.user_code)
        print("  3. approve the device; this continues automatically")
        print("\n  waiting up to %dm ..." % (dev.expires_in // 60), flush=True)

    def blocking_login():
        """Poll until approved. Blocking waits belong in a terminal, not the
        library - the web app drives poll_login() from its own loop instead."""
        dev = client.begin_login()
        on_prompt(dev)
        while not dev.expired:
            state = client.poll_login(dev)
            if state is not None:
                return state
            print("\r  waiting ... %ds left   " % dev.seconds_left,
                  end="", flush=True)
            time.sleep(dev.interval + 1)
        raise errors.LoginTimeout("device code expired before it was approved")

    try:
        state = blocking_login()
    except errors.LoginTimeout as exc:
        print("\n  TIMED OUT: %s - nothing was saved." % exc)
        return 1
    except errors.AuthError as exc:
        # Caught here rather than by main()'s handler, which would print
        # "run login" at someone who is already running login.
        print("\n  LOGIN FAILED: %s" % exc)
        return 1

    print("\n")
    _print_auth(state)
    return 0


def cmd_logout(client, _args):
    client.logout()
    print("  token cleared")
    return 0


def cmd_resolve(client, args):
    _banner("RESOLVE  %s" % args.url)
    info = client.resolve(args.url)
    print("  id            : %s" % info.id)
    print("  title         : %s" % info.title)
    print("  artist        : %s" % info.artist)
    print("  album         : %s" % info.album)
    print("  duration      : %ss" % info.duration)
    print("  explicit      : %s" % info.explicit)
    return 0


def cmd_get(client, args):
    _banner("DOWNLOAD  %s" % args.url)
    # A segmented (MPD) stream has no Content-Length up front, so total is 0
    # and a percentage is meaningless - show bytes instead of a frozen 0%.
    last = [-1.0]

    def on_progress(prog):
        done_mib = prog.downloaded / 2 ** 20
        if prog.total:
            pct = int(prog.fraction * 100)
            if pct != last[0]:
                last[0] = pct
                print("\r  %3d%%  %.1f / %.1f MiB   "
                      % (pct, done_mib, prog.total / 2 ** 20),
                      end="", flush=True)
        elif done_mib - last[0] >= 1.0:
            last[0] = done_mib
            print("\r  %.1f MiB downloaded   " % done_mib,
                  end="", flush=True)

    result = client.download(args.url, args.out,
                             quality=Quality(args.quality),
                             progress=on_progress)
    print("\n")
    print("  track         : %s" % result.track.display)
    print("  quality       : %s  codec=%s" % (result.quality, result.codec))
    print("  size          : %.1f MiB" % (result.size_bytes / 2 ** 20))
    print("  path          : %s" % result.path)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smoke_test.tidal_cli")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")

    p_login = sub.add_parser("login")
    p_login.add_argument("--force", action="store_true")

    sub.add_parser("logout")

    p_res = sub.add_parser("resolve")
    p_res.add_argument("url")

    p_get = sub.add_parser("get")
    p_get.add_argument("url")
    p_get.add_argument("-o", "--out", default="data/staging/manual")
    p_get.add_argument("-q", "--quality", default=Quality.BEST.value,
                       choices=[q.value for q in Quality])

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    handlers = {
        "status": cmd_status, "login": cmd_login,
        "logout": cmd_logout, "resolve": cmd_resolve, "get": cmd_get,
    }
    client = TidalClient(config_dir=args.config_dir)
    try:
        return handlers[args.cmd](client, args)
    except errors.AuthError as exc:
        print("  NOT AUTHENTICATED: %s" % exc)
        print("  run:  python -m smoke_test.tidal_cli login")
        return 1
    except errors.TidalError as exc:
        print("  %s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
