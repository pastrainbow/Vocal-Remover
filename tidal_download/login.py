"""The device-code login, driven for you on a background thread.

begin_login() and poll_login() on TidalClient are the primitives: they hand
back a code and check it once. Driving them is a loop, and in a terminal that
loop is three lines (see smoke_test.tidal_cli). Anywhere that cannot block it
is not, for two reasons:

  * one poll_login() call sleeps for Tidal's own poll interval (2-5s) before
    it answers, so polling it from a request handler or a UI thread ties that
    thread up for the whole login;
  * the login outlives whatever started it. A page reload, a second window, or
    a user who wanders off mid-approval must not strand it or start a second
    code.

So LoginFlow runs the loop on a daemon thread and answers `status()` instantly
from whatever it last saw. A web app asks for a code once and then polls an
endpoint that costs nothing; a desktop UI can poll it on a timer.

One flow per client, reached as `client.login`. begin() hands back the running
login rather than starting a second, because a TidalClient holds a single
pending login and a second code would silently invalidate the first.
"""
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

from . import errors
from .models import DeviceLogin

if TYPE_CHECKING:  # avoids a cycle: client.py owns the LoginFlow
    from .client import TidalClient

logger = logging.getLogger("tidal_download")


class LoginStage(str, Enum):
    """Where a login has got to.

    Distinct from AuthState, which answers a different question: whether a
    stored session works. This is about one trip through the device flow.
    """

    #: Nothing in flight - never started, or cleared by logout().
    IDLE = "idle"
    #: A code is waiting to be approved. Only this stage has one to show.
    PENDING = "pending"
    OK = "ok"
    #: The code was not approved before it expired. Start another.
    EXPIRED = "expired"
    #: Tidal refused, or the network failed. `detail` says what happened.
    ERROR = "error"


@dataclass(frozen=True)
class LoginStatus:
    """What to show about a login right now.

    The code and URL are present only while PENDING: a stale code sends
    someone to a page that cannot work.
    """

    stage: LoginStage
    detail: str = ""
    verification_url: Optional[str] = None
    user_code: Optional[str] = None
    seconds_left: int = 0

    @property
    def pending(self) -> bool:
        return self.stage is LoginStage.PENDING


class LoginFlow:
    """Owns the one in-flight device login for a client, and its thread.

    Built by TidalClient, not by callers: reach it as `client.login`.
    """

    def __init__(self, client: "TidalClient"):
        self._client = client
        self._guard = threading.Lock()
        self._stage = LoginStage.IDLE
        self._device: Optional[DeviceLogin] = None
        self._detail = ""
        #: Bumped whenever a login starts or is abandoned. A polling thread
        #: carries the value it started with, so a thread left over from a
        #: login nobody wants any more (see reset()) stops instead of
        #: reporting - or worse, saving a session after a logout.
        self._token = 0

    # ---------------------------------------------------------------- public

    def begin(self) -> LoginStatus:
        """Start a login and return its code, or hand back the live one.

        Raises AuthError if Tidal will not issue a code.
        """
        with self._guard:
            if self._live():
                return self._snapshot()

            # Inside the lock on purpose: this is one quick request, and two
            # callers racing here would each get a code, only one of which
            # the client would then be able to poll.
            device = self._client.begin_login()
            self._device = device
            self._stage = LoginStage.PENDING
            self._detail = "waiting for you to approve the code"
            self._token += 1
            threading.Thread(target=self._poll_until_done,
                             args=(device, self._token),
                             name="tidal-login", daemon=True).start()
            logger.info("login started, code %s valid for %ds",
                        device.user_code, device.expires_in)
            return self._snapshot()

    def status(self) -> LoginStatus:
        """Where the login has got to. Reads state; never talks to Tidal."""
        with self._guard:
            return self._snapshot()

    def reset(self) -> None:
        """Forget any login, in flight or finished.

        Called by TidalClient.logout(): the session a pending code would
        produce is no longer wanted, and a stale OK would leave a caller
        claiming a session that has just been deleted.
        """
        with self._guard:
            self._token += 1
            self._stage = LoginStage.IDLE
            self._device = None
            self._detail = ""

    # ----------------------------------------------------------------- work

    def _poll_until_done(self, device: DeviceLogin, token: int) -> None:
        try:
            while not device.expired:
                # Checked before each poll rather than after: poll_login()
                # saves the session as soon as the code is approved, so a
                # logout during the poll itself can still bring one back.
                # That window is one poll interval wide and needs a logout to
                # land inside it.
                if not self._current(token):
                    logger.info("abandoning the pending login")
                    return
                # Blocks for the poll interval each time round; that is the
                # whole reason this runs on a thread of its own.
                state = self._client.poll_login(device)
                if state is not None:
                    self._finish(token, LoginStage.OK, state.detail)
                    logger.info("login complete: %s", state.detail)
                    return
            self._finish(token, LoginStage.EXPIRED,
                         "the code expired before it was approved")
        except errors.LoginTimeout as exc:
            self._finish(token, LoginStage.EXPIRED, str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            logger.warning("login failed: %s: %s", type(exc).__name__, exc)
            self._finish(token, LoginStage.ERROR,
                         f"{type(exc).__name__}: {exc}")

    def _finish(self, token: int, stage: LoginStage, detail: str) -> None:
        with self._guard:
            if token != self._token:
                return  # superseded, most likely by a logout
            self._stage = stage
            self._detail = detail
            self._device = None

    def _current(self, token: int) -> bool:
        with self._guard:
            return token == self._token

    # --------------------------------------------------------------- helpers

    def _live(self) -> bool:
        """A pending login still worth showing. Call with the lock held."""
        return (self._stage is LoginStage.PENDING
                and self._device is not None
                and not self._device.expired)

    def _snapshot(self) -> LoginStatus:
        """Call with the lock held."""
        device = self._device if self._live() else None
        return LoginStatus(
            stage=self._stage,
            detail=self._detail,
            verification_url=device.verification_url if device else None,
            user_code=device.user_code if device else None,
            seconds_left=device.seconds_left if device else 0,
        )
