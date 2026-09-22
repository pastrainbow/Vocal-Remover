"""Pre-download the UVR models. A helper, not an entry point.

    ./run.sh --models

Models download themselves on first use, so this only decides *when* you wait
for the ~0.7 GiB: now, or during the first job. Already-cached models are
left alone.

Run by run.sh inside the venv, after the packages are installed - it needs
audio-separator, and it writes into data/models, which is where
vocal_remove.ModelConfig looks by default.

Two things make running this *before* starting the server worth the trouble,
rather than letting the first startup do it:

  * audio-separator's download_file_if_not_exists() streams straight to the
    final path and guards only with os.path.isfile(). An interrupted download
    therefore leaves a truncated file that every later run treats as cached,
    so the model never loads again until someone deletes it by hand. The
    cleanup below is the whole reason this is not just a bare loop.
  * a download inside Worker.start() happens while an unattended deploy is
    counting down its health check, which is how that interruption tends to
    get produced in the first place.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vocal_remove import (DEFAULT_MDX_MODEL, DEFAULT_MDXC_MODEL,  # noqa: E402
                          MDXCModelConfig)


def _wanted() -> list:
    """The models the server will actually preload.

    Falls back to the library defaults if settings will not import, so this
    stays usable on a half-built environment - which is exactly when someone
    runs it.
    """
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from app.config import get_settings

        names = list(get_settings().preload_models)
        if names:
            return names
    except Exception as exc:  # noqa: BLE001 - a helper, never a blocker
        print(f"  [warn] could not read preload_models ({type(exc).__name__}: "
              f"{exc}); falling back to defaults")
    return [DEFAULT_MDXC_MODEL, DEFAULT_MDX_MODEL]


def _snapshot(model_dir: Path) -> set:
    return {p for p in model_dir.rglob("*") if p.is_file()}


def main() -> int:
    from audio_separator.separator import Separator

    # Ask the config where models go rather than repeating the path here.
    model_dir = MDXCModelConfig().model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  into {model_dir}")

    separator = Separator(model_file_dir=str(model_dir), log_level=40)
    failed = []
    for name in _wanted():
        before = _snapshot(model_dir)
        try:
            separator.download_model_and_data(name)
            print(f"  [ok  ] {name}")
        except BaseException as exc:  # noqa: BLE001
            # BaseException, not Exception: Ctrl-C mid-download is the most
            # likely way to produce a truncated file, and it is the case
            # least forgivable to leave behind.
            for path in _snapshot(model_dir) - before:
                print(f"  [warn] removing partial {path.name}")
                path.unlink(missing_ok=True)
            if isinstance(exc, KeyboardInterrupt):
                raise
            failed.append(name)
            print(f"  [warn] {name}: {type(exc).__name__}: {exc}")

    if failed:
        print(f"  {len(failed)} model(s) not cached; they will download on "
              f"first use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
