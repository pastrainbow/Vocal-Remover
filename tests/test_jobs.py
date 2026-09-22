"""Job lifecycle tests.

Deliberately stdlib-only. app.db and app.jobs import nothing outside the
standard library, so these run on a bare CI runner with no venv, no CUDA and
no lockfile sync - which is the only part of this app a GitHub-hosted runner
can honestly verify. Anything touching torch, onnxruntime or Tidal belongs on
the self-hosted GPU box, via setup/verify_env.py.
"""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import db, jobs  # noqa: E402


class JobsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = db.init_db(Path(self._tmp.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _make(self, **kw):
        return jobs.create(self.conn, url="https://tidal.com/track/1",
                           model="m.ckpt", output_format="FLAC", **kw)


class TestSchema(JobsTestCase):
    def test_version_is_stamped(self):
        self.assertEqual(db.schema_version(self.conn), db.SCHEMA_VERSION)


class TestLifecycle(JobsTestCase):
    def test_happy_path(self):
        job = self._make()
        self.assertIs(job.stage, jobs.Stage.QUEUED)

        job = jobs.advance(self.conn, job.id, jobs.Stage.DOWNLOADING)
        self.assertIsNotNone(job.started_at)

        jobs.advance(self.conn, job.id, jobs.Stage.SEPARATING)
        job = jobs.advance(self.conn, job.id, jobs.Stage.DONE,
                           vocals_path="v.flac", instrumental_path="i.flac")

        self.assertIs(job.stage, jobs.Stage.DONE)
        self.assertEqual(job.progress, 1.0)
        self.assertIsNotNone(job.finished_at)
        self.assertEqual(job.vocals_path, "v.flac")

    def test_terminal_stages_are_final(self):
        job = jobs.advance(self.conn, self._make().id, jobs.Stage.FAILED,
                           error="boom")
        with self.assertRaises(jobs.IllegalTransition):
            jobs.advance(self.conn, job.id, jobs.Stage.DOWNLOADING)

    def test_cannot_skip_download(self):
        job = self._make()
        with self.assertRaises(jobs.IllegalTransition):
            jobs.advance(self.conn, job.id, jobs.Stage.SEPARATING)

    def test_advance_unknown_job(self):
        with self.assertRaises(KeyError):
            jobs.advance(self.conn, "nope", jobs.Stage.DOWNLOADING)


class TestQueue(JobsTestCase):
    def _backdate(self, job, created_at):
        """Force distinct timestamps.

        jobs._now() is whole seconds, so two jobs created in the same second
        tie on created_at and next_queued's ORDER BY resolves the tie by
        rowid - incidentally, not by contract. Pinning the values here tests
        the ordering the query actually promises instead of that accident.
        """
        with self.conn:
            self.conn.execute("UPDATE jobs SET created_at=? WHERE id=?",
                              (created_at, job.id))

    def test_next_queued_is_fifo(self):
        first = self._make()
        second = self._make()
        self._backdate(first, 1000)
        self._backdate(second, 2000)
        self.assertEqual(jobs.next_queued(self.conn).id, first.id)

        jobs.advance(self.conn, first.id, jobs.Stage.DOWNLOADING)
        self.assertEqual(jobs.next_queued(self.conn).id, second.id)

    def test_empty_queue(self):
        self.assertIsNone(jobs.next_queued(self.conn))


class TestDedupe(JobsTestCase):
    def test_find_active_matches_on_key(self):
        job = self._make(track_id=42)
        found = jobs.find_active(self.conn, track_id=42, model="m.ckpt",
                                 output_format="FLAC")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, job.id)

        # A different model is a different unit of work, not a duplicate.
        self.assertIsNone(jobs.find_active(self.conn, track_id=42,
                                           model="other.onnx",
                                           output_format="FLAC"))

    def test_finished_job_is_not_active(self):
        job = self._make(track_id=42)
        jobs.advance(self.conn, job.id, jobs.Stage.FAILED, error="boom")
        self.assertIsNone(jobs.find_active(self.conn, track_id=42,
                                           model="m.ckpt",
                                           output_format="FLAC"))


if __name__ == "__main__":
    unittest.main()
