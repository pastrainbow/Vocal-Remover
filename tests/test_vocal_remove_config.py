"""own_fields() and read_applied() on the config classes.

read_applied() is the reason the model settings page can show a real number
instead of an empty "model default" box, and MDXC's override of it is the
one place where the obvious reading of the instance is wrong. Both are worth
pinning.

Deliberately stdlib-only, like test_jobs.py: vocal_remove.config has no
third-party imports, and the loaded model instances these read are stubbed
with SimpleNamespace rather than loaded for real - a GPU and ~0.7 GiB of
checkpoints is not something CI has. The attribute names stubbed below are
audio-separator 0.47.0's; setup/verify_env.py is where that meets a real one.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocal_remove.config import (  # noqa: E402
    DemucsModelConfig,
    DemucsSeparationConfig,
    MDXCModelConfig,
    MDXCSeparationConfig,
    MDXModelConfig,
    MDXSeparationConfig,
    VRModelConfig,
    VRSeparationConfig,
)


def _mdxc_instance(**kw):
    """An MDXC instance as audio-separator leaves it after load_model()."""
    defaults = dict(
        segment_size=256,               # what arch_params() handed over
        override_model_segment_size=False,
        pitch_shift=0,
        overlap=8,                      # resolved in __init__ from the model
        batch_size=1,
        model_data_cfgdict=SimpleNamespace(
            inference=SimpleNamespace(dim_t=352)),  # the model's own
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestOwnFields(unittest.TestCase):
    def test_model_config_excludes_the_shared_base_fields(self):
        own = MDXCModelConfig.own_fields()
        self.assertEqual(own, ("segment_size", "pitch_shift"))
        for shared in ("name", "model_dir", "use_autocast", "log_level"):
            self.assertNotIn(shared, own)

    def test_separation_config_excludes_the_shared_base_fields(self):
        own = MDXCSeparationConfig.own_fields()
        self.assertEqual(own, ("overlap", "batch_size"))
        for shared in ("output_dir", "output_format", "vocals_name"):
            self.assertNotIn(shared, own)

    def test_architectures_without_per_job_overrides_have_none(self):
        self.assertEqual(VRSeparationConfig.own_fields(), ())
        self.assertEqual(DemucsSeparationConfig.own_fields(), ())

    def test_each_architecture_declares_its_own_set(self):
        self.assertEqual(MDXModelConfig.own_fields(),
                         ("segment_size", "hop_length", "enable_denoise"))
        self.assertIn("window_size", VRModelConfig.own_fields())
        self.assertIn("shifts", DemucsModelConfig.own_fields())


class TestMDXCReadApplied(unittest.TestCase):
    """The trap: instance.segment_size is not what MDXC runs with."""

    def test_inherited_segment_size_reports_the_models_own_dim_t(self):
        applied = MDXCModelConfig().read_applied(_mdxc_instance())
        self.assertEqual(applied["segment_size"], 352)

    def test_inherited_segment_size_is_not_the_handed_over_value(self):
        # 256 is what arch_params() passes and what the attribute holds, but
        # demix() ignores it while override_model_segment_size is False.
        applied = MDXCModelConfig().read_applied(_mdxc_instance())
        self.assertNotEqual(applied["segment_size"], 256)

    def test_overridden_segment_size_reports_the_override(self):
        instance = _mdxc_instance(segment_size=128,
                                  override_model_segment_size=True)
        applied = MDXCModelConfig(segment_size=128).read_applied(instance)
        self.assertEqual(applied["segment_size"], 128)

    def test_overlap_and_batch_size_come_straight_off_the_instance(self):
        # MDXC resolves a None for these during __init__, so unlike
        # segment_size the attribute is already the real thing.
        applied = MDXCSeparationConfig().read_applied(_mdxc_instance())
        self.assertEqual(applied, {"overlap": 8, "batch_size": 1})


class TestReadAppliedElsewhere(unittest.TestCase):
    def test_mdx_reads_every_field_off_the_instance(self):
        instance = SimpleNamespace(segment_size=256, hop_length=1024,
                                   enable_denoise=False, overlap=0.25,
                                   batch_size=1)
        self.assertEqual(
            MDXModelConfig().read_applied(instance),
            {"segment_size": 256, "hop_length": 1024, "enable_denoise": False})
        self.assertEqual(MDXSeparationConfig().read_applied(instance),
                         {"overlap": 0.25, "batch_size": 1})

    def test_vr_reads_every_field_off_the_instance(self):
        instance = SimpleNamespace(window_size=320, aggression=10,
                                   batch_size=4, enable_tta=True,
                                   enable_post_process=False,
                                   post_process_threshold=0.2,
                                   high_end_process=False)
        applied = VRModelConfig().read_applied(instance)
        self.assertEqual(applied["window_size"], 320)
        self.assertEqual(applied["aggression"], 10)
        self.assertTrue(applied["enable_tta"])

    def test_a_config_with_no_own_fields_reads_nothing(self):
        self.assertEqual(VRSeparationConfig().read_applied(object()), {})


if __name__ == "__main__":
    unittest.main()
