"""Fit theta (latency) and theta_energy on the base grid, evaluate on validation points.

    python calibrate.py   -> results/theta.json, prints error table
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq

RES = "results"
CSV = os.path.join(RES, "measurements.csv")
META = os.path.join(RES, "meta.json")
OUT = os.path.join(RES, "theta.json")
LAYER_CSV = os.path.join(RES, "layer_times.csv")
T_BIG = 100e-6  # layers slower than this are GPU-bound (launch overhead negligible)


def load(path=CSV):
    df = pd.read_csv(path)
    for c in ("memory_bytes", "latency_s", "energy_J"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_validation"] = df["is_validation"].astype(str).str.lower().eq("true")
    return df


def ape(pred, meas):
    return np.abs(np.asarray(pred) - np.asarray(meas)) / np.asarray(meas)


def stats(pred, meas):
    e = ape(pred, meas)
    if len(e) == 0:
        return {}
    return {"n": int(len(e)), "MAPE_%": float(100 * e.mean()), "median_APE_%": float(100 * np.median(e)),
            "max_APE_%": float(100 * e.max())}


# ---------------------------------------------------------------------------
# step 1: P and BW from per-layer GPU times of large (GPU-bound) layers.
# End-to-end latency alone cannot separate them: at large S*B both conv time
# (F/P) and elementwise time (Q/BW) scale as B*S^2.
#   P  = sum F / sum t over conv layers      (time-weighted effective FLOP/s)
#   BW = sum Q / sum t over bn/relu/pool     (effective DRAM bandwidth)
# ---------------------------------------------------------------------------
def layer_table(lt):
    coef = {l[0]: l[2:] for l in eq.LAYERS}
    f_s, f_c, q_s, q_c, q_w = (np.array([coef[n][i] for n in lt.layer]) for i in range(5))
    S2, B = lt.S.values.astype(float) ** 2, lt.B.values.astype(float)
    lt = lt.copy()
    lt["F"] = B * (f_s * S2 + f_c)
    lt["Q"] = B * (q_s * S2 + q_c) + q_w
    return lt


def _rate(g, col):
    return float(g[col].sum() / g.time_s.sum())


def fit_hw(lt):
    """Global P, BW + per-conv P_l; a conv gets a second P for B > B_switch if its
    efficiency changes by > 1.5x across some batch threshold (cuDNN algorithm switch)."""
    big = lt[(~lt.is_validation) & (lt.time_s > T_BIG)]
    conv = big[big.kind == "conv"]
    mem = big[big.kind.isin(["bn", "relu", "pool"])]
    P, BW = _rate(conv, "F"), _rate(mem, "Q")

    P_layer, P_big, B_switch = {}, {}, None
    for name, g in conv.groupby("layer", sort=False):
        P_layer[name] = _rate(g, "F")
        best = None
        for b in sorted(g.B.unique())[:-1]:
            lo, hi = g[g.B <= b], g[g.B > b]
            if len(lo) < 3 or len(hi) < 3:
                continue
            p_lo, p_hi = _rate(lo, "F"), _rate(hi, "F")
            sse = (np.log(lo.F / p_lo / lo.time_s) ** 2).sum() + (np.log(hi.F / p_hi / hi.time_s) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, b, p_lo, p_hi)
        if best and max(best[2], best[3]) / min(best[2], best[3]) > 1.5:
            if B_switch is None or best[1] == B_switch:
                B_switch = int(best[1])
                P_layer[name], P_big[name] = best[2], best[3]

    per_layer = {}
    for name, g in big.groupby("layer", sort=False):
        per_layer[name] = {"TFLOPs": _rate(g, "F") / 1e12, "GBs": _rate(g, "Q") / 1e9, "n": int(len(g))}
    refined = {"P": P, "BW": BW, "P_layer": P_layer, "P_layer_bigB": P_big,
               "B_switch": B_switch if B_switch is not None else float("inf")}
    return P, BW, refined, per_layer


# ---------------------------------------------------------------------------
# step 2: t0, t_op from end-to-end latency with the P's and BW fixed
# (joint 4-param fit if no layer times). Nonlinear LSQ on log T, log-params.
# ---------------------------------------------------------------------------
def fit_latency(tr, hw=None):
    S, B, T = tr.S.values, tr.B.values, tr.latency_s.values
    t_min = T.min()
    fixed = hw is not None

    def theta_of(p):
        e = np.exp(p)
        if fixed:
            return {**hw, "t0": e[0], "t_op": e[1]}
        return {"t0": e[0], "t_op": e[1], "P": e[2], "BW": e[3]}

    def res(p):
        return np.log(eq.latency(S, B, theta_of(p))) - np.log(T)

    starts = [(None, None)] if fixed else [(a, b) for a in (2e12, 4e12, 7e12) for b in (150e9, 250e9)]
    best = None
    for P0, BW0 in starts:
        p0 = [0.2 * t_min, 0.8 * t_min / eq.N_OPS] + ([] if fixed else [P0, BW0])
        r = least_squares(res, np.log(p0), method="trf", loss="soft_l1", f_scale=0.1, max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    th = {k: (float(v) if np.isscalar(v) else v) for k, v in theta_of(best.x).items()}
    th.update(k_smooth=eq.K_SMOOTH, n_ops=eq.N_OPS,
              method="P,BW from per-layer times; t0,t_op from e2e" if fixed else "joint 4-param e2e fit")
    return th


# ---------------------------------------------------------------------------
# energy: E = P_static*T + e_flop*F + e_byte*Q, non-negative LSQ on relative error
# regressor T = measured latency (physical fit); prediction uses the latency model
# ---------------------------------------------------------------------------
def fit_energy(tr, theta_lat):
    S, B, E = tr.S.values, tr.B.values, tr.energy_J.values
    A = np.column_stack([tr.latency_s.values, eq.flops(S, B), eq.bytes_moved(S, B)])
    w = 1.0 / E
    coef, _ = nnls(A * w[:, None], np.ones_like(E))
    return {"P_static": float(coef[0]), "e_flop": float(coef[1]), "e_byte": float(coef[2]),
            "latency": theta_lat}


def lat_metrics(th, sub):
    return stats(eq.latency(sub.S.values, sub.B.values, th), sub.latency_s.values)


def main():
    df = load()
    ok = df[df.status == "ok"].copy()
    ok = ok[np.isfinite(ok.latency_s) & np.isfinite(ok.energy_J) & (ok.energy_J > 0)]
    tr, va = ok[~ok.is_validation], ok[ok.is_validation]
    print(f"configs: {len(df)}  ok: {len(ok)}  fit: {len(tr)}  validation: {len(va)}  "
          f"OOM/ERR: {int((df.status != 'ok').sum())}")

    per_layer, th_simple = None, None
    if os.path.exists(LAYER_CSV):
        lt = pd.read_csv(LAYER_CSV)
        lt["is_validation"] = lt["is_validation"].astype(str).str.lower().eq("true")
        P, BW, refined, per_layer = fit_hw(layer_table(lt))
        th_simple = fit_latency(tr, {"P": P, "BW": BW})
        th_lat = fit_latency(tr, refined)
    else:
        th_lat = fit_latency(tr)
    th_en = fit_energy(tr, th_lat)

    meta = json.load(open(META)) if os.path.exists(META) else {}
    cap = meta.get("gpu_total_memory_bytes", np.inf)

    metrics = {}
    for name, sub in (("fit", tr), ("validation", va), ("all", ok)):
        S, B = sub.S.values, sub.B.values
        mem_pred, mem_meas = eq.memory(S, B), sub.memory_bytes.values
        metrics[name] = {
            "latency": lat_metrics(th_lat, sub),
            "latency_simple": lat_metrics(th_simple, sub) if th_simple else {},
            "energy": stats(eq.energy(S, B, th_en), sub.energy_J.values),
            "energy_with_measured_T": stats(
                th_en["P_static"] * sub.latency_s.values + th_en["e_flop"] * eq.flops(S, B)
                + th_en["e_byte"] * eq.bytes_moved(S, B), sub.energy_J.values),
            "memory": stats(mem_pred, mem_meas),
            "memory_exact_within_0.5%": int((ape(mem_pred, mem_meas) < 0.005).sum()),
            "memory_measured_below_model": int((mem_meas < 0.995 * mem_pred).sum()),
            "flops_vs_counter": stats(eq.flops_matmul(S, B), sub.flops_counted.values),
        }

    # OOM consistency: predicted memory vs device capacity
    pred_mem_all = eq.memory(df.S.values, df.B.values)
    oom = df.status != "ok"
    metrics["oom"] = {
        "measured_oom": int(oom.sum()),
        "predicted_oom": int((pred_mem_all > cap).sum()),
        "agree": int(((pred_mem_all > cap) == oom).sum()),
        "total": int(len(df)),
        "max_predicted_bytes": float(pred_mem_all.max()),
        "max_measured_bytes": float(np.nanmax(df.memory_bytes.values)),
        "capacity_bytes": float(cap),
    }

    theta = {"latency": th_lat, "latency_simple": th_simple, "energy": th_en, "metrics": metrics,
             "per_layer_efficiency": per_layer, "gpu": meta.get("gpu"), "idle_power_W": meta.get("idle_power_W")}
    os.makedirs(RES, exist_ok=True)
    json.dump(theta, open(OUT, "w"), indent=2, default=float)

    for title, th in (("simple (global P)", th_simple), ("refined (per-layer P)", th_lat)):
        if not th:
            continue
        print(f"\ntheta_latency {title}:")
        print(f"  t0   = {th['t0'] * 1e6:8.2f} us    (fixed per-forward overhead)")
        print(f"  t_op = {th['t_op'] * 1e6:8.2f} us    (per-op launch/dispatch floor, {eq.N_OPS} ops)")
        print(f"  P    = {th['P'] / 1e12:8.3f} TFLOP/s  BW = {th['BW'] / 1e9:.1f} GB/s")
        for n, p in th.get("P_layer", {}).items():
            extra = ""
            if n in th.get("P_layer_bigB", {}):
                extra = f"   (B > {th['B_switch']}: {th['P_layer_bigB'][n] / 1e12:.3f})"
            print(f"    P_{n:6s} = {p / 1e12:7.3f} TFLOP/s{extra}")
    if per_layer:
        print("\nper-layer effective throughput (GPU-bound points):")
        for n, v in per_layer.items():
            print(f"    {n:8s} {v['TFLOPs']:7.3f} TFLOP/s  {v['GBs']:7.1f} GB/s  (n={v['n']})")
    print("\ntheta_energy:")
    print(f"  P_static = {th_en['P_static']:7.2f} W      (idle measured: {meta.get('idle_power_W', 'n/a')})")
    print(f"  e_flop   = {th_en['e_flop'] * 1e12:7.3f} pJ/FLOP")
    print(f"  e_byte   = {th_en['e_byte'] * 1e12:7.3f} pJ/B")
    print("\nerrors (APE %):")
    for split in ("fit", "validation"):
        for q in ("latency_simple", "latency", "energy", "energy_with_measured_T", "memory", "flops_vs_counter"):
            m = metrics[split][q]
            if m:
                print(f"  {split:10s} {q:22s} MAPE={m['MAPE_%']:7.2f}  median={m['median_APE_%']:7.2f}  "
                      f"max={m['max_APE_%']:7.2f}  (n={m['n']})")
        print(f"  {split:10s} memory exact (<0.5%): {metrics[split]['memory_exact_within_0.5%']}, "
              f"measured below model: {metrics[split]['memory_measured_below_model']}")
    print("\nOOM:", metrics["oom"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
