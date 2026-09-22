"""Supervision tests: what happens when the separator process dies.

Deliberately stdlib-only, like test_jobs. app.vocal_remove_worker.supervise
imports nothing outside the standard library for exactly this reason - the
child it supervises brings in torch, but the supervising is plain process
handling and is the part most worth checking, because every path through it
starts with a crash that is awkward to arrange on purpose.

So these tests supervise a stub child instead of the real one: the module
functions below are handed to Worker in place of supervise._child_main, and
they crash, hang or behave on demand. multiprocessing pickles a target by
name, so they have to live at module level - and this module has to stay
importable by a spawned interpreter, which is why it imports nothing heavy.
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import db, jobs  # noqa: E402
from app.vocal_remove_worker import supervise  # noqa: E402

#: A model report shaped the way the real child sends one.
MODEL = {"name": "stub.ckpt", "status": "loaded", "device": "cuda",
         "load_seconds": 0.1, "error": None}

#: Upper bound on how long a stub child hangs around waiting to be stopped.
#: A test that leaves one behind should fail, not wedge the suite.
CHILD_MAX_SECONDS = 30.0

#: How long a test waits for the supervisor to notice something. Generous:
#: every restart here is a real interpreter starting up.
SETTLE_SECONDS = 30.0


@dataclass
class StubSettings:
    """What Worker actually uses of Settings - and picklable, as the real one
    has to be, since it is an argument to the spawned process."""

    db_path: Path

    def ensure_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------ stub children


def _spawns(settings: StubSettings) -> int:
    """Record this spawn and return how many there have been.

    A file rather than a counter: each child is a fresh interpreter, so the
    only memory they share with each other is the disk.
    """
    log = settings.db_path.parent / "spawns"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("x")
    return len(log.read_text(encoding="utf-8"))


def _serve(commands):
    """Idle the way the real child does: on the command pipe, until STOP."""
    deadline = time.monotonic() + CHILD_MAX_SECONDS
    while time.monotonic() < deadline:
        if commands.poll(0.05) and commands.recv() == supervise.STOP:
            return


def child_ready(settings, report, commands):
    """A healthy worker: reports its models, then waits to be stopped."""
    _spawns(settings)
    report.send((supervise.READY, [MODEL]))
    _serve(commands)


def child_fatal(settings, report, commands):
    """A worker that cannot run - the models would not load."""
    _spawns(settings)
    report.send((supervise.FATAL, "ModelsUnavailable: stub.ckpt: no such file"))


def child_ready_then_broken(settings, report, commands):
    """Healthy the first time, then dies on every restart.

    The shape of a GPU that has gone away: the worker was fine until it was
    not, and nothing that comes after it can start.
    """
    if _spawns(settings) == 1:
        report.send((supervise.READY, [MODEL]))
        _serve(commands)
    else:
        os._exit(1)


def child_hangs(settings, report, commands):
    """Never reports ready, never exits - a model load wedged on the network."""
    _spawns(settings)
    time.sleep(CHILD_MAX_SECONDS)


def child_ignores_stop(settings, report, commands):
    """Ready, then deaf to STOP - as a separation in flight is."""
    _spawns(settings)
    report.send((supervise.READY, [MODEL]))
    time.sleep(CHILD_MAX_SECONDS)


# -------------------------------------------------------------------- tests


def _wait_until(predicate, timeout=SETTLE_SECONDS):
    """Poll until the supervisor has done something, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class WorkerTestCase(unittest.TestCase):
    #: Which stub this case's worker spawns. Overridden per subclass.
    child = staticmethod(child_ready)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = StubSettings(db_path=Path(self._tmp.name) / "test.db")
        db.init_db(self.settings.db_path).close()

        # Real backoff is seconds; these tests would otherwise spend all of
        # them waiting. The cap on consecutive failures is kept honest by
        # deriving it the same way the module does.
        self._saved = (supervise._BACKOFF_SECONDS,
                       supervise._MAX_CONSECUTIVE_FAILURES,
                       supervise._READY_TIMEOUT_SECONDS,
                       supervise._child_main)
        supervise._BACKOFF_SECONDS = (0.01,) * 3
        supervise._MAX_CONSECUTIVE_FAILURES = len(supervise._BACKOFF_SECONDS)
        supervise._child_main = self.child

        self.worker = supervise.Worker(self.settings)

    def tearDown(self):
        try:
            self.worker.stop(timeout=2.0)
        finally:
            (supervise._BACKOFF_SECONDS, supervise._MAX_CONSECUTIVE_FAILURES,
             supervise._READY_TIMEOUT_SECONDS, supervise._child_main) = self._saved
            self._tmp.cleanup()

    def conn(self):
        return db.connect(self.settings.db_path)

    def queue_separating_job(self) -> str:
        """A job in the stage a crash would interrupt."""
        conn = self.conn()
        try:
            job = jobs.create(conn, url="https://tidal.com/track/1",
                              model="stub.ckpt", output_format="FLAC")
            jobs.advance(conn, job.id, jobs.Stage.DOWNLOADING)
            jobs.advance(conn, job.id, jobs.Stage.SEPARATING)
            return job.id
        finally:
            conn.close()

    def fetch(self, job_id: str) -> jobs.Job:
        conn = self.conn()
        try:
            return jobs.get(conn, job_id)
        finally:
            conn.close()


class TestStartup(WorkerTestCase):
    def test_start_waits_for_the_models(self):
        self.worker.start()
        self.assertTrue(self.worker.running)
        self.assertEqual(self.worker.loaded_models, ("stub.ckpt",))
        self.assertEqual(self.worker.status()["models"], [MODEL])
        self.assertIsNotNone(self.worker.status()["pid"])

    def test_start_fails_jobs_left_by_the_last_run(self):
        job_id = self.queue_separating_job()
        self.worker.start()
        job = self.fetch(job_id)
        self.assertIs(job.stage, jobs.Stage.FAILED)
        self.assertIn("restart", job.error)


class TestStartupRefusal(WorkerTestCase):
    child = staticmethod(child_fatal)

    def test_a_child_that_cannot_run_aborts_startup(self):
        with self.assertRaises(supervise.WorkerStartupError) as caught:
            self.worker.start()
        self.assertIn("no such file", str(caught.exception))
        self.assertFalse(self.worker.running)


class TestHang(WorkerTestCase):
    child = staticmethod(child_hangs)

    def test_a_child_that_never_reports_is_not_waited_on_forever(self):
        supervise._READY_TIMEOUT_SECONDS = 0.5
        with self.assertRaises(supervise.WorkerStartupError) as caught:
            self.worker.start()
        self.assertIn("did not finish loading", str(caught.exception))


class TestRestart(WorkerTestCase):
    def test_a_killed_worker_comes_back(self):
        self.worker.start()
        first = self.worker.status()["pid"]

        self.worker._process.kill()

        # Not just `running`: for a moment after the kill the supervisor has
        # not looked yet, and the worker still looks up. It is a different
        # process being up that says the restart happened.
        self.assertTrue(_wait_until(
            lambda: self.worker.running
            and self.worker.status()["pid"] not in (None, first)))
        self.assertEqual(self.worker.status()["restarts"], 1)
        self.assertEqual(self.worker.loaded_models, ("stub.ckpt",))

    def test_the_job_it_was_running_is_failed(self):
        self.worker.start()
        job_id = self.queue_separating_job()

        self.worker._process.kill()

        self.assertTrue(_wait_until(
            lambda: self.fetch(job_id).stage is jobs.Stage.FAILED))
        # The error says the worker died, not that somebody restarted the
        # server - that difference is the whole point of recording it here.
        self.assertIn("stopped unexpectedly", self.fetch(job_id).error)

    def test_notifying_a_dead_worker_does_not_block(self):
        """A regression test with a bug behind it.

        wake and stop were multiprocessing.Events until a child killed while
        idle inside Event.wait() left the shared Condition owed an
        acknowledgement it would never get - and the next notify(), on a
        request thread, blocked forever. Pipes have no such bookkeeping.
        """
        self.worker.start()
        self.worker._process.kill()

        returned = threading.Event()

        def submit():
            self.worker.notify()
            returned.set()

        threading.Thread(target=submit, daemon=True).start()
        self.assertTrue(returned.wait(10.0),
                        "notify() blocked on a worker that had been killed")

    def test_status_explains_the_gap_while_it_is_down(self):
        self.worker.start()
        self.worker._process.kill()

        self.assertTrue(_wait_until(
            lambda: self.worker.status()["restarts"] == 1))
        # Whether this catches the restarting window or the worker already
        # back up, status() must be answerable throughout: a health check
        # during a restart is exactly when one gets asked.
        self.assertIsInstance(self.worker.status()["detail"], str)
        self.assertTrue(_wait_until(lambda: self.worker.running))
        self.assertEqual(self.worker.status()["detail"], "running")


class TestGivingUp(WorkerTestCase):
    child = staticmethod(child_ready_then_broken)

    def test_a_worker_that_cannot_restart_stops_trying(self):
        self.worker.start()
        self.worker._process.kill()

        attempts = supervise._MAX_CONSECUTIVE_FAILURES
        self.assertTrue(_wait_until(
            lambda: self.worker.status()["restarts"] > attempts))
        # One failure per attempt plus the original kill, and then it stops:
        # a crash loop against a broken model file is not worth the log.
        self.assertFalse(_wait_until(
            lambda: self.worker.status()["restarts"] > attempts + 1,
            timeout=2.0))
        self.assertFalse(self.worker.running)
        self.assertIn("times in a row", self.worker.status()["detail"])


class TestStop(WorkerTestCase):
    child = staticmethod(child_ignores_stop)

    def test_a_busy_worker_is_terminated_rather_than_waited_out(self):
        self.worker.start()
        proc = self.worker._process

        started = time.monotonic()
        self.worker.stop(timeout=0.5)

        self.assertLess(time.monotonic() - started, SETTLE_SECONDS)
        self.assertFalse(proc.is_alive())
        self.assertFalse(self.worker.running)

    def test_stopping_does_not_restart_it(self):
        self.worker.start()
        self.worker.stop(timeout=0.5)
        self.assertEqual(self.worker.status()["restarts"], 0)
        self.assertEqual(self.worker.status()["detail"], "stopped")


class TestExplainExit(unittest.TestCase):
    """The log line a crash produces is the only evidence there will be."""

    def test_clean_exit(self):
        self.assertIn("cleanly", supervise._explain_exit(0))

    def test_plain_failure(self):
        self.assertEqual(supervise._explain_exit(3), "exited with code 3")

    def test_windows_fault_is_named(self):
        self.assertIn("access violation", supervise._explain_exit(0xC0000005))

    def test_unknown_windows_status_is_at_least_legible(self):
        self.assertIn("0xC0000022", supervise._explain_exit(0xC0000022))

    def test_signal_is_named(self):
        self.assertIn("killed by", supervise._explain_exit(-15))

    def test_oom_kill_says_so(self):
        # Only meaningful where SIGKILL exists; elsewhere the number stands.
        text = supervise._explain_exit(-9)
        self.assertIn("9" if sys.platform == "win32" else "OOM", text)


if __name__ == "__main__":
    unittest.main()
