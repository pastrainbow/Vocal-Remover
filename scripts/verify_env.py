"""Health check for the application environment.

Operational utility, not a dev script: setup.ps1 runs it after installing, and
doctor.ps1 runs it on its own. Exits 0 if everything the app needs is working,
1 otherwise, so it is usable from CI or a pre-flight check.

It deliberately checks behaviour rather than presence - notably it builds a
real ONNX Runtime session instead of trusting get_available_providers(), which
reports CUDA even when CUDA cannot initialise.
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(name):
    def wrap(fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - report, never crash the run
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        RESULTS.append((ok, name, detail))
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {name:<22} {detail}")
        return fn

    return wrap


print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)


@check("python")
def _python():
    v = sys.version_info
    ok = (v.major, v.minor) == (3, 11)
    return ok, f"{v.major}.{v.minor}.{v.micro}  {sys.executable}"


# torch MUST be imported before onnxruntime: it registers the CUDA DLL
# directory that ORT's provider library depends on. Reordering these silently
# drops every .onnx model to CPU. See requirements/app.in.
@check("torch + CUDA")
def _torch():
    import torch

    if not torch.cuda.is_available():
        return False, f"{torch.__version__} - CUDA NOT AVAILABLE (CPU-only wheel?)"
    name = torch.cuda.get_device_name(0)
    _, total = torch.cuda.mem_get_info()
    return True, f"{torch.__version__}  {name}  {total / 2 ** 30:.1f} GiB"


@check("onnxruntime CUDA")
def _ort():
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    g = helper.make_graph(
        [helper.make_node("MatMul", ["A", "B"], ["C"])], "g",
        [helper.make_tensor_value_info("A", TensorProto.FLOAT, [8, 8]),
         helper.make_tensor_value_info("B", TensorProto.FLOAT, [8, 8])],
        [helper.make_tensor_value_info("C", TensorProto.FLOAT, [8, 8])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 10
    probe = ROOT / "_ort_probe.onnx"
    probe.write_bytes(m.SerializeToString())
    try:
        s = ort.InferenceSession(str(probe), providers=["CUDAExecutionProvider"])
        provs = s.get_providers()
    finally:
        probe.unlink(missing_ok=True)
    if "CUDAExecutionProvider" not in provs:
        return False, f"{ort.__version__} - fell back to CPU, .onnx models will be ~10x slow"
    return True, f"{ort.__version__}  session={provs[0]}"


@check("ffmpeg")
def _ffmpeg():
    import shutil

    path = shutil.which("ffmpeg")
    return bool(path), path or "not on PATH - audio decode/encode will fail"


@check("audio-separator")
def _sep():
    from audio_separator.separator import Separator  # noqa: F401
    import importlib.metadata as md

    return True, md.version("audio-separator")


@check("models cached")
def _models():
    d = ROOT / "data" / "models"
    if not d.is_dir():
        return True, "none yet (downloaded on first separation)"
    files = [f for f in d.iterdir() if f.suffix in (".ckpt", ".onnx", ".pth")]
    size = sum(f.stat().st_size for f in files) / 2 ** 30
    if not files:
        return True, "none yet (downloaded on first separation)"
    return True, f"{len(files)} model(s), {size:.2f} GiB"


@check("tidal_download")
def _tidal():
    from tidal_download import TidalClient

    state = TidalClient().auth_status()
    if state.valid:
        return True, f"authenticated as {state.user_id} ({state.country_code})"
    # Not an environment failure - the env is fine, you just have not logged in.
    return True, f"NOT LOGGED IN ({state.detail}) - run the login command below"


@check("web stack")
def _web():
    import fastapi
    import uvicorn

    return True, f"fastapi {fastapi.__version__}, uvicorn {uvicorn.__version__}"


failed = [r for r in RESULTS if not r[0]]
print("=" * 70)
if failed:
    print(f"FAILED ({len(failed)}):")
    for _, name, detail in failed:
        print(f"  - {name}: {detail}")
    print("=" * 70)
    sys.exit(1)

try:
    from tidal_download import TidalClient

    logged_in = TidalClient().auth_status().valid
except Exception:
    logged_in = False

print("ENVIRONMENT OK")
if not logged_in:
    print("")
    print("Next step - log in to Tidal (one time, opens a browser link):")
    print(r"  venv\Scripts\python.exe -m smoke_test.tidal_cli login")
print("=" * 70)
sys.exit(0)
