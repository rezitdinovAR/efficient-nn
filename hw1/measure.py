"""Measure latency / peak memory / energy of SmallCNN on the (S, B) grid.

    python measure.py              # full grid -> results/measurements.csv, layer_times.csv, meta.json
    python measure.py --resume     # continue after a Colab disconnect
    python measure.py --quick      # smoke test on a few points
"""
import argparse
import gc
import json
import math
import os
import platform
import threading
import time

import numpy as np
import pandas as pd
import torch

import equations as eq
from models import build_model

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 2026
RES = "results"
CSV = os.path.join(RES, "measurements.csv")
META = os.path.join(RES, "meta.json")
LAYER_CSV = os.path.join(RES, "layer_times.csv")


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------
def make_grid(seed=SEED):
    rng = np.random.default_rng(seed)
    cand_s = [s for s in range(32, 513, 16) if s not in BASE_S]
    cand_b = [b for b in range(1, 257) if b & (b - 1)]          # not a power of two
    rand_s = sorted(int(v) for v in rng.choice(cand_s, 4, replace=False))
    rand_b = sorted(int(v) for v in rng.choice(cand_b, 3, replace=False))
    all_s, all_b = sorted(BASE_S + rand_s), sorted(BASE_B + rand_b)
    grid = [(s, b, (s in rand_s) or (b in rand_b)) for s in all_s for b in all_b]
    return grid, rand_s, rand_b


# ---------------------------------------------------------------------------
# energy via NVML: hardware energy counter (Volta+), else power sampling
# ---------------------------------------------------------------------------
class EnergyMeter:
    def __init__(self, index=0, period=0.005):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.period = period
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.mode = "counter"
        except pynvml.NVMLError:
            self.mode = "sampling"

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0

    def _sample(self):
        while not self._stop.is_set():
            self._samples.append((time.perf_counter(), self.power_w()))
            time.sleep(self.period)

    def start(self):
        if self.mode == "counter":
            self._e0 = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h)  # mJ
        else:
            self._samples, self._stop = [], threading.Event()
            self._th = threading.Thread(target=self._sample, daemon=True)
            self._th.start()

    def stop(self):
        """Joules since start()."""
        if self.mode == "counter":
            return (self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) - self._e0) / 1000.0
        self._stop.set()
        self._th.join()
        t, p = np.array(self._samples).T
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(p, t)) if len(t) > 1 else float("nan")

    def state(self):
        nv = self.nv
        return {"sm_clock_MHz": nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_SM),
                "mem_clock_MHz": nv.nvmlDeviceGetClockInfo(self.h, nv.NVML_CLOCK_MEM),
                "temp_C": nv.nvmlDeviceGetTemperature(self.h, nv.NVML_TEMPERATURE_GPU)}

    def idle_power(self, seconds=3.0):
        ps = []
        t_end = time.time() + seconds
        while time.time() < t_end:
            ps.append(self.power_w())
            time.sleep(0.05)
        return float(np.median(ps))


# ---------------------------------------------------------------------------
# FLOPs via torch's counter on the meta device (no GPU memory, works for OOM cases)
# ---------------------------------------------------------------------------
def count_flops(meta_model, S, B):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.empty(B, 3, S, S, device="meta")
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        meta_model(x)
    return int(fc.get_total_flops())


# ---------------------------------------------------------------------------
# per-layer GPU time with CUDA events (used to identify P and BW separately)
# ---------------------------------------------------------------------------
LAYER_NAMES = [l[0] for l in eq.LAYERS]
LAYER_KINDS = [l[1] for l in eq.LAYERS]


def layer_times(model, x, reps=3):
    layers = [m for m in model if not isinstance(m, torch.nn.Flatten)]
    flat_at = [i for i, m in enumerate(model) if isinstance(m, torch.nn.Flatten)][0]
    n = len(layers)
    assert n == eq.N_OPS
    res = np.zeros((reps, n))
    for r in range(reps):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
        h = x
        ev[0].record()
        for i, layer in enumerate(layers):
            h = layer(h)
            if i == flat_at - 1:
                h = torch.flatten(h, 1)
            ev[i + 1].record()
        torch.cuda.synchronize()
        res[r] = [ev[i].elapsed_time(ev[i + 1]) * 1e-3 for i in range(n)]
        del h
    return np.median(res, axis=0)


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------
def measure_config(model, S, B, meter, a):
    rec = {"status": "ok"}
    x = None
    try:
        x = torch.randn(B, 3, S, S, device="cuda")
        with torch.inference_mode():
            for _ in range(a.warmup):
                model(x)
            torch.cuda.synchronize()

            # --- peak memory of one forward
            torch.cuda.reset_peak_memory_stats()
            rec["mem_before_bytes"] = torch.cuda.memory_allocated()
            model(x)
            torch.cuda.synchronize()
            rec["memory_bytes"] = torch.cuda.max_memory_allocated()

            # --- latency: median of synchronized single forwards
            times, t_start = [], time.perf_counter()
            while True:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                n = len(times)
                if n >= a.max_reps or (n >= a.min_reps and time.perf_counter() - t_start >= a.target_time):
                    break
            times = np.asarray(times)
            lat = float(np.median(times))
            rec.update(latency_s=lat, latency_p10_s=float(np.percentile(times, 10)),
                       latency_p90_s=float(np.percentile(times, 90)), n_reps=int(len(times)))

            # --- energy: back-to-back forwards over a >= energy_window window
            n_e = max(a.min_energy_iters, math.ceil(a.energy_window / lat))
            torch.cuda.synchronize()
            meter.start()
            t0 = time.perf_counter()
            for _ in range(n_e):
                model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            e = meter.stop()
            rec.update(energy_J=e / n_e, t_loop_s=(t1 - t0) / n_e, power_W=e / (t1 - t0),
                       n_energy_iters=n_e, **meter.state())

            # --- per-layer GPU times
            rec["_layers"] = layer_times(model, x).tolist()
    except torch.cuda.OutOfMemoryError:
        rec = {"status": "OOM", "memory_bytes": "OOM"}
    except RuntimeError as err:  # e.g. cuDNN refusing huge tensors
        msg = str(err).splitlines()[0][:120]
        rec = {"status": "OOM" if "out of memory" in msg.lower() else f"ERR: {msg}",
               "memory_bytes": "OOM"}
    finally:
        del x
        gc.collect()
        torch.cuda.empty_cache()
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--min-reps", type=int, default=10)
    p.add_argument("--max-reps", type=int, default=200)
    p.add_argument("--target-time", type=float, default=1.0, help="s of latency sampling per config")
    p.add_argument("--energy-window", type=float, default=3.0, help="s of energy integration per config")
    p.add_argument("--min-energy-iters", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test, writes to results_quick/")
    a = p.parse_args()
    global RES, CSV, META, LAYER_CSV
    if a.quick:
        RES = "results_quick"
        CSV, META, LAYER_CSV = (os.path.join(RES, f) for f in ("measurements.csv", "meta.json", "layer_times.csv"))

    assert torch.cuda.is_available(), "no CUDA device"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    os.makedirs(RES, exist_ok=True)

    model = build_model().cuda().eval()
    meta_model = build_model().to("meta").eval()
    meter = EnergyMeter(torch.cuda.current_device())

    grid, rand_s, rand_b = make_grid()
    if a.quick:
        grid = [(32, 1, False), (128, 16, False), (224, 7, True), (512, 64, False)]
        a.target_time, a.energy_window = 0.3, 1.0

    # --- static state: weights + library workspaces after a warm-up forward
    with torch.inference_mode():
        model(torch.randn(2, 3, 64, 64, device="cuda"))
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    static_bytes = torch.cuda.memory_allocated()
    props = torch.cuda.get_device_properties(0)
    meta = {
        "gpu": props.name, "gpu_total_memory_bytes": props.total_memory,
        "sm_count": props.multi_processor_count, "compute_capability": f"{props.major}.{props.minor}",
        "driver": meter.nv.nvmlSystemGetDriverVersion(),
        "power_limit_W": meter.nv.nvmlDeviceGetPowerManagementLimit(meter.h) / 1000.0,
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(), "energy_mode": meter.mode,
        "idle_power_W": meter.idle_power(), "static_allocated_bytes": static_bytes,
        "param_alloc_bytes_predicted": eq.PARAM_ALLOC_BYTES,
        "lib_overhead_measured_bytes": static_bytes - eq.PARAM_ALLOC_BYTES,
        "lib_overhead_assumed_bytes": eq.CUBLAS_WS_BYTES,
        "random_S": rand_s, "random_B": rand_b, "seed": SEED, "settings": vars(a),
    }
    json.dump(meta, open(META, "w"), indent=2)
    print(json.dumps(meta, indent=2))

    done, rows, lrows = set(), [], []
    if a.resume and os.path.exists(CSV):
        rows = pd.read_csv(CSV).to_dict("records")
        done = {(int(r["S"]), int(r["B"])) for r in rows}
        if os.path.exists(LAYER_CSV):
            lrows = pd.read_csv(LAYER_CSV).to_dict("records")

    order = np.random.default_rng(SEED).permutation(len(grid))  # shuffled: no thermal bias vs S/B
    todo = [grid[i] for i in order if (grid[i][0], grid[i][1]) not in done]
    t_all = time.time()
    for i, (S, B, is_val) in enumerate(todo, 1):
        rec = {"S": S, "B": B, "is_validation": is_val}
        rec.update(measure_config(model, S, B, meter, a))
        for name, kind, t in zip(LAYER_NAMES, LAYER_KINDS, rec.pop("_layers", [])):
            lrows.append({"S": S, "B": B, "is_validation": is_val, "layer": name, "kind": kind, "time_s": t})
        rec.update(flops_counted=count_flops(meta_model, S, B), flops_pred=eq.flops(S, B),
                   flops_matmul_pred=eq.flops_matmul(S, B), memory_pred=eq.memory(S, B),
                   bytes_pred=eq.bytes_moved(S, B))
        rows.append(rec)
        pd.DataFrame(rows).sort_values(["S", "B"]).to_csv(CSV, index=False)
        if lrows:
            pd.DataFrame(lrows).to_csv(LAYER_CSV, index=False)
        lat = rec.get("latency_s", float("nan"))
        mem = f"{rec['memory_bytes'] / 2**20:9.1f}" if rec["status"] == "ok" else f"{rec['status']:>9s}"
        print(f"[{i:3d}/{len(todo)}] S={S:3d} B={B:3d} {'val' if is_val else 'fit'}  "
              f"T={lat * 1e3:9.3f} ms  M={mem} MiB (pred {rec['memory_pred'] / 2**20:9.1f})  "
              f"E={rec.get('energy_J', float('nan')):.4f} J   ({time.time() - t_all:.0f}s)", flush=True)
    print("saved", CSV)


if __name__ == "__main__":
    main()
