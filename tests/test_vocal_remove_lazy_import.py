"""vocal_remove/__init__.py's lazy import of .separator.

Runs each check in a fresh subprocess rather than in-process: unittest
discover loads every test module into one interpreter, so an in-process
`'torch' not in sys.modules` assertion here would be at the mercy of import
order across the whole suite (once any test touches vr.init_models/separate,
torch is loaded for the rest of the process). A subprocess makes each
assertion self-contained.

Deliberately stdlib-only, like test_jobs.py: subprocess and sys are all this
needs, and vocal_remove.config itself has no third-party dependencies - see
vocal_remove/__init__.py's docstring for why that matters here specifically.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(SRC), capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    return result.stdout.strip()


class TestLazySeparatorImport(unittest.TestCase):
    def test_importing_config_does_not_load_torch(self):
        out = _run(
            "import sys\n"
            "import vocal_remove.config\n"
            "print('torch' in sys.modules)\n"
        )
        self.assertEqual(out, "False")

    def test_importing_the_package_does_not_load_torch(self):
        out = _run(
            "import sys\n"
            "import vocal_remove\n"
            "print('torch' in sys.modules)\n"
        )
        self.assertEqual(out, "False")

    def test_touching_a_config_class_does_not_load_torch(self):
        out = _run(
            "import sys\n"
            "import vocal_remove as vr\n"
            "vr.ModelConfig(name='foo.onnx', model_dir='models')\n"
            "vr.SeparationConfig(output_dir='out')\n"
            "print('torch' in sys.modules)\n"
        )
        self.assertEqual(out, "False")

    # There is no matching "touching init_models DOES load torch" test here:
    # that needs audio_separator installed, which - like torch itself - this
    # stdlib-only suite deliberately cannot assume (see test_jobs.py's
    # docstring). setup/verify_env.py covers that positive case on the GPU
    # box, where the real dependencies exist.

    def test_unknown_attribute_still_raises_attribute_error(self):
        with self.assertRaises(AssertionError):
            _run(
                "import vocal_remove as vr\n"
                "vr.this_does_not_exist\n"
            )


if __name__ == "__main__":
    unittest.main()
