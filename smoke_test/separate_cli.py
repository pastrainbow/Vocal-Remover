"""Manual smoke-test CLI for vocal_remove.

DEV SCRIPT - not imported by the web app. A thin shell over init_models() and
separate(); it holds no logic of its own.

    venv\\Scripts\\python.exe -m smoke_test.separate_cli <audio> [-o out_dir]
    venv\\Scripts\\python.exe -m smoke_test.separate_cli <audio> -m UVR-MDX-NET-Voc_FT.onnx
    venv\\Scripts\\python.exe -m smoke_test.separate_cli <audio> --segment 128
"""
import argparse
import logging
import sys
from pathlib import Path

from vocal_remove import (
    DEFAULT_MODEL,
    MDXCModelConfig,
    MDXCSeparationConfig,
    MDXModelConfig,
    MDXSeparationConfig,
    errors,
    init_models,
    separate,
)

#: The architecture is chosen by file extension. audio-separator itself works
#: it out from the model data, but we must pick the matching config subclass
#: so the parameters land in the dict that architecture actually reads.
_BY_EXTENSION = {
    ".onnx": (MDXModelConfig, MDXSeparationConfig),
    ".ckpt": (MDXCModelConfig, MDXCSeparationConfig),
    ".yaml": (MDXCModelConfig, MDXCSeparationConfig),
}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="smoke_test.separate_cli")
    parser.add_argument("audio")
    parser.add_argument("-o", "--out", default="data/out/manual")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-f", "--format", default="FLAC")
    parser.add_argument("--segment", type=int, default=None,
                        help="load-time; lower this (e.g. 128) on CUDA OOM")
    parser.add_argument("--overlap", type=float, default=None,
                        help="per-job; higher is slower, cleaner at seams")
    parser.add_argument("--batch", type=int, default=None,
                        help="per-job; trades VRAM for throughput")
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    print("=" * 66)
    print("SEPARATE  %s" % args.audio)
    print("=" * 66)

    model_cls, sep_cls = _BY_EXTENSION.get(
        Path(args.model).suffix.lower(), (MDXCModelConfig, MDXCSeparationConfig))

    model = init_models([model_cls(
        name=args.model,
        segment_size=args.segment,
        use_autocast=not args.no_autocast,
    )])[0]

    print("  model         : %s  (%s)" % (args.model, model_cls.__name__))
    print("  status        : %s" % model.status.value)
    if not model.ok:
        print("  MODEL LOAD FAILED: %s" % model.error)
        return 1
    print("  device        : %s  (loaded in %.1fs)" % (model.device, model.load_seconds))
    if not model.on_gpu:
        print("  WARNING       : not on GPU - this will be very slow")

    try:
        stems = separate(args.audio, model,
                         sep_cls(output_dir=args.out,
                                 output_format=args.format,
                                 overlap=args.overlap,
                                 batch_size=args.batch))
    except errors.OutOfMemory as exc:
        print("  CUDA OOM: %s" % exc)
        return 1
    except (errors.AudioNotFound, errors.SeparationError) as exc:
        print("  %s: %s" % (type(exc).__name__, exc))
        return 1

    def mib(p):
        return Path(p).stat().st_size / 2 ** 20

    print("  separation    : %.1fs" % stems.seconds)
    print("  vocals        : %s  (%.1f MiB)" % (stems.vocals, mib(stems.vocals)))
    print("  instrumental  : %s  (%.1f MiB)" % (stems.instrumental,
                                                mib(stems.instrumental)))
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
