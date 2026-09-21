"""The Tidal device-code login, driven from a background thread.

Tidal's flow is: ask for a code, show it to the user, then poll until they
approve it in a browser. Two things make that awkward to run inside request
handlers:

  * one TidalClient.poll_login() call sleeps for Tidal's own poll interval
    (2-5s) before it answers, so polling from the page would tie up a request
    handler for the whole login;
  * the login outlives the page. A reload, a second tab, or a user who wanders
    off mid-approval must not strand it or start a second code.

So the polling lives here on one thread and the routes only read state off it:
the page asks for a code once, then polls a status endpoint that answers
instantly and survives a refresh.

One flow at a time, per process. begin() hands back the running one rather
than starting a second, because TidalClient keeps a single pending login and
a second code would silently invalidate the one already on screen.
"""
import logging
import threading
from enum import Enum
from typing import Optional

import tidal_download as td

logger = logging.getLogger("app.auth")


class LoginState(str, Enum):
    """Where a login has got to. Only PENDING has a code to show."""

    IDLE = "idle"
    PENDING = "pending"
    OK = "ok"
    #: The code was not approved before it expired. Start another.
    EXPIRED = "expired"
    #: Tidal refused, or the network failed. `detail` says what happened.
    ERROR = "error"


class LoginFlow:
    """Owns the one in-flight device login, and its polling thread."""

    def __init__(self, client: td.TidalClient):
        self._client = client
        self._guard = threading.Lock()
        self._state = LoginState.IDLE
        self._device: Optional[td.DeviceLogin] = None
        self._detail = ""
        #: Bumped whenever a login starts or is abandoned. A polling thread
        #: carries the value it started with, so a thread left over from a
        #: login nobody wants any more (see reset()) stops instead of
        #: reporting - or worse, saving a session after a sign-out.
        self._token = 0

    # ---------------------------------------------------------------- public

    def begin(self) -> dict:
        """Start a login and return its code, or hand back the live one.

        Raises td.AuthError if Tidal will not issue a code.
        """
        with self._guard:
            if self._live():
                return self._snapshot()

            # Inside the lock on purpose: this is one quick request, and two
            # callers racing here would each get a code, only one of which
            # TidalClient would then be able to poll.
            device = self._client.begin_login()
            self._device = device
            self._state = LoginState.PENDING
            self._detail = "waiting for you to approve the code"
            self._token += 1
            threading.Thread(target=self._poll_until_done,
                             args=(device, self._token),
                             name="tidal-login", daemon=True).start()
            logger.info("tidal login started, code %s valid for %ds",
                        device.user_code, device.expires_in)
            return self._snapshot()

    def status(self) -> dict:
        with self._guard:
            return self._snapshot()

    def reset(self) -> None:
        """Forget any login, in flight or finished.

        Called on sign-out: the session a pending code would produce is no
        longer wanted, and a stale OK would leave the page claiming a session
        that has just been deleted.
        """
        with self._guard:
            self._token += 1
            self._state = LoginState.IDLE
            self._device = None
            self._detail = ""

    # ----------------------------------------------------------------- work

    def _poll_until_done(self, device: td.DeviceLogin, token: int) -> None:
        try:
            while not device.expired:
                # Checked before each poll rather than after: poll_login()
                # saves the session as soon as the code is approved, so a
                # sign-out during the poll itself can still bring one back.
                # That window is one poll interval wide and needs a sign-out
                # to land inside it.
                if not self._current(token):
                    logger.info("abandoning the pending tidal login")
                    return
                # Blocks for the poll interval each time round; that is the
                # whole reason this is a thread and not a request handler.
                state = self._client.poll_login(device)
                if state is not None:
                    self._finish(token, LoginState.OK, state.detail)
                    logger.info("tidal login complete: %s", state.detail)
                    return
            self._finish(token, LoginState.EXPIRED,
                         "the code expired before it was approved")
        except td.LoginTimeout as exc:
            self._finish(token, LoginState.EXPIRED, str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.warning("tidal login failed: %s: %s",
                           type(exc).__name__, exc)
            self._finish(token, LoginState.ERROR, f"{type(exc).__name__}: {exc}")

    def _finish(self, token: int, state: LoginState, detail: str) -> None:
        with self._guard:
            if token != self._token:
                return  # superseded, most likely by a sign-out
            self._state = state
            self._detail = detail
            self._device = None

    def _current(self, token: int) -> bool:
        with self._guard:
            return token == self._token

    # --------------------------------------------------------------- helpers

    def _live(self) -> bool:
        """A pending login still worth showing. Call with the lock held."""
        return (self._state is LoginState.PENDING
                and self._device is not None
                and not self._device.expired)

    def _snapshot(self) -> dict:
        """Call with the lock held."""
        device = self._device if self._live() else None
        return {
            "state": self._state.value,
            "detail": self._detail,
            # Only while pending: the URL carries the code, and a stale one
            # would send someone to a page that cannot work.
            "verification_url": device.verification_url if device else None,
            "user_code": device.user_code if device else None,
            "seconds_left": device.seconds_left if device else 0,
        }
