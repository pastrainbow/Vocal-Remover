"""Supervises the separator process.

Separation runs in a child process, not a thread - see process.py for the
other half. This module is the parent half: it spawns that process, waits for
it to report its models, restarts it when it dies, and says why in the log
and in /api/health.

The reason for the split is fate sharing. Separation is the one part of this
app that can take the interpreter down without an exception first: a CUDA
fault, a bad free inside a native library, or the host OOM killer choosing the
process holding two models resident. On a thread, every one of those took the
web server with it. In a child process they cost the job that was running and
a few seconds of restart.

What the parent keeps: the database, which was never shared memory to begin
with, and the model report, so /api/health can still say what the worker had
loaded while it is down.

Imports here are stdlib only, and that is load-bearing twice over. torch and
audio-separator arrive through process.py, which is imported inside
_child_main() and therefore only ever in the child - so the web process never
pays for two gigabytes of weights or a CUDA context. And supervision is the
part of this app most worth testing and least convenient to reach on a GPU
box, so tests/test_worker.py drives it with a stub child on a bare runner.
Adding `from .. import pipeline` at the top of this file would undo both.
"""
import logging
import multiprocessing as mp
import signal
import sqlite3
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple

from .. import db, jobs as jobs_repo

if TYPE_CHECKING:  # pydantic is not worth importing to name a parameter
    from ..config import Settings

#: Both halves log under this one name rather than their module paths: a
#: reader watching one terminal wants "the worker", and which process a line
#: came from is the supervisor's job to say.
logger = logging.getLogger("app.worker")

#: What the child says, up its own pipe. It sends one of these once, and then
#: either works or exits.
READY = "ready"
FATAL = "fatal"

#: What the parent says, down a second pipe going the other way.
WAKE = "wake"
STOP = "stop"

# Two pipes rather than multiprocessing.Event, which would read better and is
# unusable here: Event.set() takes a shared Condition, and Condition.notify()
# waits for every sleeper it wakes to acknowledge. A child killed inside
# Event.wait() - which is where an idle worker spends its life, and killed is
# how this one dies - never acknowledges, so the next set() in the web process
# blocks forever. Pipes carry no shared state to corrupt: a dead peer is a
# closed handle, which reads as EOF and writes as an error.

#: How long the child gets to load its models before the parent gives up on
#: it. Warm loads take seconds; the first ever start downloads the weights,
#: which is why this is minutes rather than seconds.
_READY_TIMEOUT_SECONDS = 300.0

#: How often the supervisor wakes to look at the pipe and the process. It is
#: a poll rather than a blocking wait on both because Connection.poll and
#: Process.is_alive behave the same on Windows and POSIX, and
#: connection.wait() does not.
_SUPERVISOR_POLL_SECONDS = 0.5

#: Backoff between restarts, by consecutive failure. A crash that repeats
#: immediately is usually a broken model file or a GPU that has gone away, and
#: hammering either just fills the log.
_BACKOFF_SECONDS = (1.0, 2.0, 5.0, 15.0, 30.0)
_MAX_CONSECUTIVE_FAILURES = len(_BACKOFF_SECONDS)

#: A child that survives this long is treated as having started successfully,
#: so the next crash begins a fresh run of the backoff. Without it a server up
#: for a week would exhaust its restarts on unrelated crashes months apart.
_HEALTHY_SECONDS = 120.0

#: Exit statuses Windows reports for a process that faulted rather than
#: exited. Worth naming: "0xC0000005" in a log is a question, "access
#: violation" is an answer.
_WINDOWS_FAULTS = {
    0xC0000005: "access violation",
    0xC000001D: "illegal instruction",
    0xC0000017: "out of memory",
    0xC0000094: "integer division by zero",
    0xC00000FD: "stack overflow",
    0xC0000374: "heap corruption",
    0xC0000409: "stack buffer overrun",
}


class WorkerStartupError(RuntimeError):
    """The separator process would not start.

    Raised from start(), which makes it a refusal to bring the server up.
    Starting anyway would only turn one clear error into a stream of failed
    jobs - the models are preloaded precisely so that this is decided once.
    """


def _child_main(settings: "Settings", report, commands) -> None:
    """Entry point for the spawned process.

    The import is deliberately inside the function. multiprocessing pickles
    this target by name, so the child imports this module and calls it - and
    only then pulls in torch, audio-separator and the rest, in the process
    that is meant to own them.
    """
    from . import process

    process.run(settings, report, commands)


class Worker:
    """The separator process, and the thread that keeps it alive."""

    def __init__(self, settings: "Settings"):
        self.settings = settings

        self._process: Optional[mp.process.BaseProcess] = None
        #: Child to parent: the model report. Read by the supervisor only.
        self._pipe = None
        #: Parent to child: wake and stop. Written by request handlers and by
        #: the supervisor, so every send goes through the lock.
        self._commands = None
        self._sending = threading.Lock()

        #: Set once, by stop(). Distinguishes a shutdown from a crash, and
        #: cuts a restart backoff short.
        self._shutdown = threading.Event()
        self._supervisor: Optional[threading.Thread] = None

        self._model_report: List[dict] = []
        self._loaded: Tuple[str, ...] = ()
        self._ready = False
        self._ready_deadline = 0.0
        self._started_at = 0.0

        #: Consecutive failed starts, and the total restarts since start().
        self._failures = 0
        self._restarts = 0
        #: Why the child is about to die, when that is already known - a kill
        #: this process ordered, or a FATAL the child sent on its way out.
        #: Preferred over the exit status, which would only say "killed".
        self._pending_reason: Optional[str] = None
        self._detail = "not started"

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Bring up the worker, or raise saying why it cannot come up.

        Blocks until the child reports its models, so the first request meets
        a healthy worker rather than racing it.
        """
        self.settings.ensure_dirs()

        # NB: sqlite3's context manager commits, it does not close - so this
        # is deliberately explicit rather than a `with` block.
        conn = db.init_db(self.settings.db_path)
        try:
            orphaned = jobs_repo.reset_orphans(conn)
        finally:
            conn.close()
        if orphaned:
            logger.warning("failed %d job(s) left mid-flight by a restart",
                           orphaned)

        self._shutdown.clear()
        self._failures = 0
        self._restarts = 0

        # spawn, not fork: CUDA does not survive a fork, and it is the only
        # start method Windows has anyway - so the child is a fresh
        # interpreter on both.
        self._spawn(mp.get_context("spawn"))

        error = self._await_ready()
        if error is not None:
            self.stop()
            raise WorkerStartupError(f"the separator process would not "
                                     f"start: {error}")

        self._supervisor = threading.Thread(target=self._supervise,
                                            name="worker-supervisor",
                                            daemon=True)
        self._supervisor.start()
        logger.info("worker process %s ready with %d model(s): %s",
                    self._process.pid, len(self._loaded),
                    ", ".join(self._loaded))

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the worker to finish and go.

        A job already running is given `timeout` to finish, which it will not
        if it is mid-separation: the loop only reads its commands between
        jobs. Waiting out a separation would hold shutdown for minutes, so the
        process is terminated instead and the job is failed as an orphan by
        the next start().
        """
        self._shutdown.set()
        self._command(STOP)
        self._await_exit(timeout)

        supervisor = self._supervisor
        if supervisor is not None:
            supervisor.join(timeout=_SUPERVISOR_POLL_SECONDS * 4)
        self._supervisor = None

        # The supervisor can have been inside _spawn() when the shutdown was
        # set, in which case there is now a child younger than the one joined
        # above. Nothing is ever going to ask it for anything.
        self._await_exit(0.0)

        self._close_pipes()
        self._process = None
        self._ready = False
        self._detail = "stopped"

    def notify(self) -> None:
        """Wake the worker immediately - call after queueing a job.

        Best effort by design. A nudge lost to a worker that is restarting as
        it is sent costs the job one idle poll, and the queue is in the
        database either way: nothing depends on this arriving.
        """
        self._command(WAKE)

    # --------------------------------------------------------------- status

    @property
    def running(self) -> bool:
        """Whether a job submitted now would be picked up."""
        proc = self._process
        return bool(self._ready and proc is not None and proc.is_alive())

    @property
    def loaded_models(self) -> Tuple[str, ...]:
        """Names of the models the worker has resident.

        Names, not models: the models themselves only exist in the child.
        """
        return self._loaded

    def status(self) -> dict:
        proc = self._process
        return {
            "running": self.running,
            "pid": proc.pid if proc is not None and proc.is_alive() else None,
            "restarts": self._restarts,
            "detail": self._detail,
            "models": list(self._model_report),
        }

    # ----------------------------------------------------------- supervision

    def _spawn(self, ctx=None) -> None:
        """Start a child, its two pipes, and the clock it has to report in."""
        ctx = ctx or mp.get_context("spawn")
        self._close_pipes()
        self._pipe, report_end = ctx.Pipe(duplex=False)
        command_end, commands = ctx.Pipe(duplex=False)
        with self._sending:
            self._commands = commands

        # Not a daemon: audio-separator may run pools of its own for some
        # architectures, and a daemonic process cannot have children. stop()
        # does the job daemon=True would have done, and does it explicitly.
        proc = ctx.Process(target=_child_main, name="separator",
                           args=(self.settings, report_end, command_end))
        proc.start()
        # Each pipe has exactly one writer and one reader once these two go.
        # Dropping the ends this process does not use is what makes a death
        # on either side show up as EOF on the other - including this process
        # being killed outright, which the child treats as its cue to leave
        # rather than sitting on a GPU nobody is using.
        report_end.close()
        command_end.close()

        self._process = proc
        self._ready = False
        self._started_at = time.monotonic()
        self._ready_deadline = self._started_at + _READY_TIMEOUT_SECONDS
        self._detail = "loading models"
        logger.info("spawned worker process %s", proc.pid)

    def _await_ready(self) -> Optional[str]:
        """Block until the child reports ready. Returns None, or the reason.

        Used only by start(); once the supervisor thread is running it owns
        the pipe and does the same job without blocking anyone.
        """
        while True:
            message = self._receive()
            if message is not None:
                kind, payload = message
                if kind == READY:
                    return None
                if kind == FATAL:
                    return payload
                continue

            proc = self._process
            if proc is None:
                return "it was never started"
            if not proc.is_alive():
                return _explain_exit(proc.exitcode)
            if time.monotonic() >= self._ready_deadline:
                self._kill(proc)
                return (f"it did not finish loading within "
                        f"{_READY_TIMEOUT_SECONDS:.0f}s")

    def _supervise(self) -> None:
        """Watch the child for its lifetime, and every replacement's."""
        while not self._shutdown.is_set():
            if self._receive() is not None:
                continue

            proc = self._process
            if proc is None or self._shutdown.is_set():
                break

            if not proc.is_alive():
                if not self._restart(proc):
                    break
            elif not self._ready and time.monotonic() >= self._ready_deadline:
                logger.error("worker process %s has not reported ready in "
                             "%.0fs; restarting it", proc.pid,
                             _READY_TIMEOUT_SECONDS)
                self._pending_reason = (f"it did not finish loading within "
                                        f"{_READY_TIMEOUT_SECONDS:.0f}s")
                self._kill(proc)

        logger.info("worker supervisor exited")

    def _restart(self, proc) -> bool:
        """Handle a dead child. Returns False when it is not worth retrying.

        Everything the dead process was doing is accounted for here: the log
        gets the reason, the job it was running is failed rather than left
        looking active, and /api/health gets a sentence explaining the gap.
        """
        ran_for = time.monotonic() - self._started_at
        pid = proc.pid
        reason = self._pending_reason or _explain_exit(proc.exitcode)
        self._pending_reason = None
        self._ready = False
        # Dropped before anything slow happens below: there genuinely is no
        # worker between here and the next spawn, and a health check asked
        # during the backoff should say so rather than describe a corpse.
        self._process = None

        logger.error("worker process %s died after %.1fs: %s",
                     pid, ran_for, reason)
        self._fail_inflight(reason)

        # A child that stayed up long enough to be doing real work earns a
        # clean slate. The cap is for a worker that cannot start at all, not a
        # ration on restarts over the life of the server.
        if ran_for >= _HEALTHY_SECONDS:
            self._failures = 0
        self._failures += 1
        self._restarts += 1

        if self._failures > _MAX_CONSECUTIVE_FAILURES:
            logger.critical("the worker has died %d times in a row; giving "
                            "up. Fix the cause and restart the server.",
                            self._failures)
            self._detail = (f"down: it died {self._failures} times in a row, "
                            f"the last time because {reason}. Restart the "
                            f"server once that is fixed.")
            return False

        delay = _BACKOFF_SECONDS[self._failures - 1]
        logger.warning("restarting the worker process in %.0fs (attempt %d of "
                       "%d)", delay, self._failures, _MAX_CONSECUTIVE_FAILURES)
        self._detail = f"restarting in {delay:.0f}s after: {reason}"
        if self._shutdown.wait(delay):
            return False

        try:
            self._spawn()
        except Exception as exc:
            logger.critical("could not spawn a replacement worker process",
                            exc_info=True)
            self._detail = f"down: could not spawn a replacement ({exc})"
            return False
        return True

    def _fail_inflight(self, reason: str) -> None:
        """Fail whatever the dead process was in the middle of.

        The worker is the only thing that moves a job past `queued`, so
        anything still in an active stage belonged to the process that just
        died. Left alone it would sit at "separating" forever, with a
        progress bar that never moves.
        """
        try:
            conn = db.connect(self.settings.db_path)
            try:
                orphaned = jobs_repo.reset_orphans(
                    conn, reason=f"the worker stopped unexpectedly ({reason})")
            finally:
                conn.close()
        except sqlite3.Error:
            logger.exception("could not fail the job the worker was running")
            return
        if orphaned:
            logger.warning("failed %d job(s) the worker was running when it "
                           "died", orphaned)

    # --------------------------------------------------------------- pipes

    def _command(self, command: str) -> None:
        """Send one command to the child, if there is one to send it to.

        Never raises: the pipe of a process that has just died is exactly
        where this lands, and a caller queueing a job should not see that.
        """
        with self._sending:
            commands = self._commands
            if commands is None:
                return
            try:
                commands.send(command)
            except (BrokenPipeError, EOFError, OSError, ValueError):
                logger.debug("could not send %r to the worker", command,
                             exc_info=True)

    def _receive(self) -> Optional[Tuple[str, object]]:
        """Read one message, waiting up to a poll interval. None if there was
        none - which is also how both loops here pace themselves."""
        pipe = self._pipe
        if pipe is None:
            # No child to hear from; keep the caller's loop from spinning.
            self._shutdown.wait(_SUPERVISOR_POLL_SECONDS)
            return None
        try:
            if not pipe.poll(_SUPERVISOR_POLL_SECONDS):
                return None
            kind, payload = pipe.recv()
        except (EOFError, OSError):
            # The child closed its end, so it is on its way out. The liveness
            # check is what reacts to that; this just stops re-reading EOF.
            self._close_pipe()
            return None

        if kind == READY:
            self._model_report = list(payload)
            self._loaded = tuple(m["name"] for m in payload
                                 if m["status"] == "loaded")
            self._ready = True
            self._detail = "running"
        elif kind == FATAL:
            # The child logs its own traceback before sending this; repeating
            # it here would only double it in the terminal. It exits straight
            # afterwards, so the reason is held for the death that follows -
            # otherwise that gets reported as an unexplained clean exit.
            logger.error("worker process reported it cannot run: %s", payload)
            self._pending_reason = str(payload)
            self._detail = f"down: {payload}"
        else:
            logger.warning("ignoring unknown message from the worker: %r", kind)
        return kind, payload

    def _close_pipe(self) -> None:
        """Drop the report pipe only; the command pipe outlives a bad read."""
        self._pipe = _closed(self._pipe)

    def _close_pipes(self) -> None:
        self._close_pipe()
        with self._sending:
            self._commands = _closed(self._commands)

    # ------------------------------------------------------------- process

    def _await_exit(self, timeout: float) -> None:
        """Give the current child `timeout` to leave, then make it."""
        proc = self._process
        if proc is None or not proc.is_alive():
            return
        if timeout > 0:
            proc.join(timeout=timeout)
        if proc.is_alive():
            logger.warning("worker process %s has not stopped; terminating it",
                           proc.pid)
            self._kill(proc)

    def _kill(self, proc) -> None:
        """Terminate a child, escalating if it does not go."""
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            logger.warning("worker process %s ignored terminate; killing it",
                           proc.pid)
            proc.kill()
            proc.join(timeout=5.0)


def _closed(pipe):
    """Close a pipe end if there is one, and hand back None for it."""
    if pipe is not None:
        try:
            pipe.close()
        except OSError:
            pass
    return None


def _explain_exit(code: Optional[int]) -> str:
    """Say what an exit status means, in words worth putting in a log.

    A crash is the interesting case and the one with the least readable
    status: POSIX reports the signal as a negative number, Windows reports a
    fault as an NTSTATUS that looks like an ordinary exit code until you
    notice it is 3.2 billion.
    """
    if code is None:
        return "it is still running"
    if code == 0:
        return "it exited cleanly without being asked to"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        if name == "SIGKILL":
            return (f"killed by {name} - the host OOM killer does this when "
                    f"memory runs out")
        return f"killed by {name}"
    if code in _WINDOWS_FAULTS:
        return f"crashed: {_WINDOWS_FAULTS[code]} (0x{code:08X})"
    if code > 0xFFFF:
        return f"crashed with status 0x{code:08X}"
    return f"exited with code {code}"
