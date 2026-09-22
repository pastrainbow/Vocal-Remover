"""Model settings persistence tests.

Deliberately stdlib-only, like test_jobs.py: app.model_settings imports
nothing outside json/dataclasses, precisely so this can run on a bare CI
runner alongside the rest of tests/ - see that module's docstring.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import model_settings as ms  # noqa: E402


class ModelSettingsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestDefaults(ModelSettingsTestCase):
    def test_load_with_no_file_returns_defaults(self):
        settings = ms.load(self.state_dir)
        self.assertEqual(settings, ms.ModelSettings())

    def test_default_model_is_one_of_the_preloaded_models(self):
        settings = ms.ModelSettings()
        settings.validate()  # must not raise
        self.assertIn(settings.default_model, settings.preload_models)


class TestRoundTrip(ModelSettingsTestCase):
    def test_save_then_load_recovers_the_same_values(self):
        original = ms.ModelSettings(
            preload_models=["a.ckpt", "b.onnx"],
            default_model="b.onnx",
            output_format="WAV",
            segment_size=128,
        )
        ms.save(self.state_dir, original)
        self.assertEqual(ms.load(self.state_dir), original)

    def test_save_writes_into_the_state_dir(self):
        ms.save(self.state_dir, ms.ModelSettings())
        self.assertTrue((self.state_dir / ms.FILENAME).is_file())

    def test_save_creates_missing_state_dir(self):
        nested = self.state_dir / "not" / "yet" / "created"
        ms.save(nested, ms.ModelSettings())
        self.assertTrue((nested / ms.FILENAME).is_file())

    def test_no_tmp_file_left_behind(self):
        ms.save(self.state_dir, ms.ModelSettings())
        leftovers = list(self.state_dir.glob("*.tmp"))
        self.assertEqual(leftovers, [])


class TestValidation(ModelSettingsTestCase):
    def test_empty_preload_models_rejected(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.ModelSettings(preload_models=[]).validate()

    def test_default_model_not_in_preload_models_rejected(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.ModelSettings(preload_models=["a.ckpt"],
                             default_model="b.onnx").validate()

    def test_unknown_output_format_rejected(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.ModelSettings(output_format="OGG").validate()

    def test_zero_segment_size_rejected(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.ModelSettings(segment_size=0).validate()

    def test_negative_segment_size_rejected(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.ModelSettings(segment_size=-1).validate()

    def test_none_segment_size_is_valid(self):
        ms.ModelSettings(segment_size=None).validate()  # must not raise

    def test_save_of_invalid_settings_raises_and_does_not_write(self):
        with self.assertRaises(ms.InvalidModelSettings):
            ms.save(self.state_dir, ms.ModelSettings(preload_models=[]))
        self.assertFalse((self.state_dir / ms.FILENAME).exists())


class TestCorruptFile(ModelSettingsTestCase):
    def _write_raw(self, text: str) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ms.FILENAME).write_text(text, encoding="utf-8")

    def test_unparsable_json_falls_back_to_defaults(self):
        self._write_raw("not json at all")
        self.assertEqual(ms.load(self.state_dir), ms.ModelSettings())

    def test_invalid_values_fall_back_to_defaults(self):
        self._write_raw('{"preload_models": [], "default_model": "x", '
                        '"output_format": "FLAC", "segment_size": null}')
        self.assertEqual(ms.load(self.state_dir), ms.ModelSettings())

    def test_partial_file_fills_in_missing_fields_with_defaults(self):
        self._write_raw('{"output_format": "WAV"}')
        settings = ms.load(self.state_dir)
        self.assertEqual(settings.output_format, "WAV")
        self.assertEqual(settings.preload_models, ms.ModelSettings().preload_models)


if __name__ == "__main__":
    unittest.main()
