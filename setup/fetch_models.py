"""Pre-download the UVR models. A helper, not an entry point.

    ./run.sh --models

Models download themselves on first use, so this only decides *when* you wait
for the ~0.7 GiB: now, or during the first job. Already-cached models are
left alone.

Run by run.sh inside the venv, after the packages are installed - it needs
audio-separator, and it writes into data/models, which is where
vocal_remove.ModelConfig looks by default.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocal_remove import (DEFAULT_MDX_MODEL, DEFAULT_MDXC_MODEL,  # noqa: E402
                          MDXCModelConfig)


def main() -> int:
    from audio_separator.separator import Separator

    # Ask the config where models go rather than repeating the path here.
    model_dir = MDXCModelConfig().model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  into {model_dir}")

    separator = Separator(model_file_dir=str(model_dir), log_level=40)
    failed = []
    for name in (DEFAULT_MDXC_MODEL, DEFAULT_MDX_MODEL):
        try:
            separator.download_model_and_data(name)
            print(f"  [ok  ] {name}")
        except Exception as exc:  # noqa: BLE001 - one bad model is not fatal
            failed.append(name)
            print(f"  [warn] {name}: {type(exc).__name__}: {exc}")

    if failed:
        print(f"  {len(failed)} model(s) not cached; they will download on "
              f"first use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
