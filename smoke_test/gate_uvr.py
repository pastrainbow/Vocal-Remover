"""Phase 0a gate: prove CUDA is live and a real separation runs on GPU.

Usage:  python scripts/gate_uvr.py [audio] [--model M] [--segment N] [--no-autocast]
Exits non-zero if the environment is not GPU-capable.
"""
import argparse, os, sys, threading, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("audio", nargs="?", default=str(ROOT / "test_track.flac"))
ap.add_argument("--model", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
ap.add_argument("--segment", type=int, default=None)
ap.add_argument("--no-autocast", action="store_true")
args = ap.parse_args()

print("=" * 62)
print("PHASE 0a GATE - UVR / CUDA")
print("=" * 62)

import torch
print(f"torch                : {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"torch.cuda.available : {cuda_ok}")
if cuda_ok:
    print(f"device               : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM total / free    : {total/2**30:.2f} GiB / {free/2**30:.2f} GiB")
    print(f"arch list            : {torch.cuda.get_arch_list()}")
else:
    print("!! torch has no CUDA - this is the CPU-wheel trap. STOP.")

# NOTE: torch is imported above ON PURPOSE. It registers its CUDA DLL
# directory, without which onnxruntime cannot find cudnn64_9.dll and
# falls back to CPU silently.
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto
print(f"onnxruntime          : {ort.__version__}")
print(f"ORT providers (build): {ort.get_available_providers()}")

# get_available_providers() is the COMPILE-TIME list and is true even when
# CUDA cannot initialise. The only honest check is to build a real session.
_g = helper.make_graph([helper.make_node("MatMul", ["A", "B"], ["C"])], "g",
    [helper.make_tensor_value_info("A", TensorProto.FLOAT, [8, 8]),
     helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 8])],
    [helper.make_tensor_value_info("C", TensorProto.FLOAT, [8, 8])])
_m = helper.make_model(_g, opset_imports=[helper.make_opsetid("", 18)])
_m.ir_version = 10
_probe = ROOT / "data" / "_ort_probe.onnx"
_probe.write_bytes(_m.SerializeToString())
try:
    _s = ort.InferenceSession(str(_probe), providers=["CUDAExecutionProvider"])
    ort_ok = "CUDAExecutionProvider" in _s.get_providers()
    print(f"ORT session (actual) : {_s.get_providers()}")
except Exception as e:
    ort_ok = False
    print(f"ORT session (actual) : FAILED {type(e).__name__}")
finally:
    _probe.unlink(missing_ok=True)
if not ort_ok:
    print("!! ORT on CPU - every .onnx model (all MDX-Net) runs ~10x slow. See uvr.in.")

if not cuda_ok:
    sys.exit(2)

# ---- peak VRAM sampler (driver-level: captures torch AND onnxruntime) ----
peak = {"used": 0}
stop = threading.Event()
def sample():
    _, tot = torch.cuda.mem_get_info()
    while not stop.is_set():
        f, t = torch.cuda.mem_get_info()
        peak["used"] = max(peak["used"], t - f)
        time.sleep(0.25)
threading.Thread(target=sample, daemon=True).start()

from audio_separator.separator import Separator

mdxc = {"segment_size": 256, "override_model_segment_size": False,
        "batch_size": None, "overlap": None, "pitch_shift": 0}
if args.segment:
    mdxc["segment_size"] = args.segment
    mdxc["override_model_segment_size"] = True

sep = Separator(
    log_level=40,
    model_file_dir=str(ROOT / "data" / "models"),
    output_dir=str(ROOT / "data" / "out"),
    output_format="FLAC",
    use_autocast=not args.no_autocast,
    mdxc_params=mdxc,
)
print(f"\nffmpeg               : {'OK' if sep.check_ffmpeg_installed() is not False else 'MISSING'}")
print(f"model                : {args.model}")
print(f"autocast             : {not args.no_autocast}   segment: {mdxc['segment_size']}")

t0 = time.time(); sep.load_model(args.model); t_load = time.time() - t0
print(f"model load           : {t_load:.1f}s")

print("\nseparating ...")
t1 = time.time()
try:
    outs = sep.separate(args.audio)
except torch.cuda.OutOfMemoryError:
    stop.set()
    print("!! CUDA OOM - retry with a smaller --segment (e.g. 128)")
    sys.exit(3)
t_sep = time.time() - t1
stop.set()

import subprocess
dur = float(subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
    "-of","default=nw=1:nk=1", args.audio], capture_output=True, text=True).stdout.strip())

print("=" * 62)
print(f"separation time      : {t_sep:.1f}s  for {dur:.1f}s audio  ->  {dur/t_sep:.2f}x realtime")
print(f"peak VRAM (driver)   : {peak['used']/2**30:.2f} GiB")
print("outputs:")
for o in outs:
    p = pathlib.Path(o)
    p = p if p.is_absolute() else ROOT / "data" / "out" / p
    print(f"  {p.name}  ({p.stat().st_size/2**20:.1f} MiB)")
print("=" * 62)
print("GATE PASSED" if (cuda_ok and ort_ok) else "GATE PASSED (torch CUDA ok, ORT CPU-only)")
