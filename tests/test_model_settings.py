"""Per-model inference parameter tests.

Deliberately stdlib-only, like test_jobs.py: app.model_settings imports
vocal_remove.config for the real per-architecture field names and
validation, and that module is itself stdlib-only (dataclasses/abc/pathlib) -
see vocal_remove/__init__.py's docstring for how its heavier sibling,
.separator (torch, audio_separator), stays out of this import path.
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import model_settings as ms  # noqa: E402


class ParamsTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestArchDispatch(unittest.TestCase):
    def test_onnx_is_mdx(self):
        self.assertEqual(ms.arch_for("foo.onnx"), "mdx")

    def test_pth_is_vr(self):
        self.assertEqual(ms.arch_for("foo.pth"), "vr")

    def test_htdemucs_yaml_is_demucs(self):
        self.assertEqual(ms.arch_for("htdemucs_ft.yaml"), "demucs")

    def test_other_yaml_is_mdxc(self):
        self.assertEqual(ms.arch_for("some_mdx23c_model.yaml"), "mdxc")

    def test_ckpt_is_mdxc(self):
        self.assertEqual(ms.arch_for("foo.ckpt"), "mdxc")


class TestParamsDefaults(ParamsTestCase):
    def test_mdxc_fields_split_between_load_time_and_per_job(self):
        p = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertEqual(p.load_time_fields, ("segment_size", "pitch_shift"))
        self.assertEqual(p.per_job_fields, ("overlap", "batch_size"))
        self.assertEqual(p.values["segment_size"], None)
        self.assertEqual(p.values["pitch_shift"], 0)

    def test_vr_has_no_per_job_fields(self):
        p = ms.load_params(self.state_dir, "foo.pth")
        self.assertEqual(p.per_job_fields, ())
        self.assertIn("window_size", p.load_time_fields)

    def test_demucs_segment_size_is_text_not_number(self):
        p = ms.load_params(self.state_dir, "htdemucs_ft.yaml")
        self.assertEqual(p.kinds["segment_size"], "text")
        self.assertEqual(p.values["segment_size"], "Default")

    def test_mdxc_segment_size_is_number_not_text(self):
        p = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertEqual(p.kinds["segment_size"], "number")

    def test_bool_fields_are_kind_bool(self):
        p = ms.load_params(self.state_dir, "foo.onnx")
        self.assertEqual(p.kinds["enable_denoise"], "bool")


class TestParamsRoundTrip(ParamsTestCase):
    def test_save_then_load_recovers_the_same_values(self):
        saved = ms.save_params(self.state_dir, "foo.ckpt",
                               {"segment_size": 128, "overlap": 4})
        loaded = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertEqual(loaded.values, saved.values)
        self.assertEqual(loaded.values["segment_size"], 128)
        self.assertEqual(loaded.values["overlap"], 4)

    def test_partial_save_keeps_other_saved_fields(self):
        ms.save_params(self.state_dir, "foo.ckpt", {"segment_size": 128})
        after = ms.save_params(self.state_dir, "foo.ckpt", {"overlap": 4})
        self.assertEqual(after.values["segment_size"], 128)
        self.assertEqual(after.values["overlap"], 4)

    def test_different_models_do_not_share_state(self):
        ms.save_params(self.state_dir, "foo.ckpt", {"segment_size": 128})
        other = ms.load_params(self.state_dir, "bar.ckpt")
        self.assertIsNone(other.values["segment_size"])

    def test_no_tmp_file_left_behind(self):
        ms.save_params(self.state_dir, "foo.ckpt", {"segment_size": 128})
        leftovers = list(self.state_dir.glob("*.tmp"))
        self.assertEqual(leftovers, [])


class TestParamsValidation(ParamsTestCase):
    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ms.UnknownModelParam):
            ms.save_params(self.state_dir, "foo.ckpt", {"window_size": 512})

    def test_mdxc_overlap_below_one_is_rejected(self):
        from vocal_remove.errors import InvalidConfig
        with self.assertRaises(InvalidConfig):
            ms.save_params(self.state_dir, "foo.ckpt", {"overlap": 0})

    def test_mdx_overlap_out_of_unit_range_is_rejected(self):
        from vocal_remove.errors import InvalidConfig
        with self.assertRaises(InvalidConfig):
            ms.save_params(self.state_dir, "foo.onnx", {"overlap": 1.5})

    def test_wrong_type_is_rejected(self):
        # MDXCSeparationConfig.__post_init__ compares overlap against 1, so a
        # non-numeric value fails there with a TypeError - unlike a plain
        # dataclass field, which accepts any type with no check at all.
        with self.assertRaises(TypeError):
            ms.save_params(self.state_dir, "foo.ckpt", {"overlap": "oops"})

    def test_rejected_save_does_not_persist(self):
        with self.assertRaises(ms.UnknownModelParam):
            ms.save_params(self.state_dir, "foo.ckpt", {"window_size": 512})
        self.assertFalse((self.state_dir / ms.PARAMS_FILENAME).exists())


class TestCorruptFile(ParamsTestCase):
    def test_unparsable_json_falls_back_to_defaults(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ms.PARAMS_FILENAME).write_text("not json",
                                                          encoding="utf-8")
        p = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertEqual(p.values["segment_size"], None)

    def test_invalid_saved_value_falls_back_to_defaults(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ms.PARAMS_FILENAME).write_text(
            '{"foo.ckpt": {"overlap": 0}}', encoding="utf-8")
        p = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertIsNone(p.values["overlap"])

    def test_unknown_saved_field_is_ignored_not_fatal(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / ms.PARAMS_FILENAME).write_text(
            '{"foo.ckpt": {"segment_size": 128, "made_up": 1}}',
            encoding="utf-8")
        p = ms.load_params(self.state_dir, "foo.ckpt")
        self.assertEqual(p.values["segment_size"], 128)
        self.assertNotIn("made_up", p.values)


if __name__ == "__main__":
    unittest.main()
