"""Sweep MDXC segment_size / batch_size and measure speed, VRAM and quality drift.

Quality drift matters: override_model_segment_size=True moves the model off the
segment length it was trained at, so a speed win can cost separation quality.
We therefore diff every run's Vocals stem against the run-0 baseline.
"""
import time, threading, pathlib, sys
import torch, numpy as np, soundfile as sf
from audio_separator.separator import Separator

ROOT  = pathlib.Path(__file__).resolve().parents[1]
AUDIO = str(ROOT / "test_track.flac")
MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUT   = ROOT / "data" / "sweep"; OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("baseline seg=256 (model default)", dict(segment_size=256,  override_model_segment_size=False, batch_size=None)),
    ("seg=512  override",                dict(segment_size=512,  override_model_segment_size=True,  batch_size=None)),
    ("seg=1024 override",                dict(segment_size=1024, override_model_segment_size=True,  batch_size=None)),
    ("seg=256  batch=2",                 dict(segment_size=256,  override_model_segment_size=False, batch_size=2)),
    ("seg=256  batch=4",                 dict(segment_size=256,  override_model_segment_size=False, batch_size=4)),
]

_, TOTAL = torch.cuda.mem_get_info()
BASE_USED = TOTAL - torch.cuda.mem_get_info()[0]
print(f"idle VRAM in use: {BASE_USED/2**30:.2f} GiB of {TOTAL/2**30:.2f} GiB\n")
print(f"{'config':<34}{'sep(s)':>8}{'xRT':>8}{'VRAM GiB':>10}{'drift':>10}")
print("-" * 70)

baseline_voc = None
for name, mp in CONFIGS:
    mp = {**mp, "overlap": None, "pitch_shift": 0}
    peak = {"u": 0}; stop = threading.Event()
    def sample():
        while not stop.is_set():
            f, t = torch.cuda.mem_get_info(); peak["u"] = max(peak["u"], t - f); time.sleep(0.2)
    threading.Thread(target=sample, daemon=True).start()

    tag = name.split()[0] + "_" + str(mp["segment_size"]) + "_b" + str(mp["batch_size"])
    sep = Separator(log_level=50, model_file_dir=str(ROOT / "data" / "models"),
                    output_dir=str(OUT), output_format="WAV",
                    use_autocast=True, mdxc_params=mp)
    sep.load_model(MODEL)
    t0 = time.time()
    try:
        outs = sep.separate(AUDIO, custom_output_names={"Vocals": f"{tag}_voc",
                                                        "Instrumental": f"{tag}_inst"})
    except torch.cuda.OutOfMemoryError:
        stop.set(); print(f"{name:<34}{'OOM':>8}"); torch.cuda.empty_cache(); continue
    dt = time.time() - t0; stop.set()

    voc, _ = sf.read(str(OUT / f"{tag}_voc.wav"), dtype="float32")
    if baseline_voc is None:
        baseline_voc, drift = voc, "  (ref)"
    else:
        n = min(len(voc), len(baseline_voc))
        d = voc[:n] - baseline_voc[:n]
        rms_d, rms_b = np.sqrt((d**2).mean()), np.sqrt((baseline_voc[:n]**2).mean())
        drift = f"{20*np.log10(rms_d/rms_b + 1e-12):+7.1f}dB"
    print(f"{name:<34}{dt:>8.1f}{175.125/dt:>8.2f}{peak['u']/2**30:>10.2f}{drift:>10}")
    del sep; torch.cuda.empty_cache()
