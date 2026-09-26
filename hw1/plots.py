"""All figures: measured points + predicted curves/surfaces.

    python plots.py   -> results/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import equations as eq  # noqa: E402
from calibrate import load  # noqa: E402

RES = "results"
FIG = os.path.join(RES, "figures")
BD = np.geomspace(1, 256, 200)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)
    print("  ", name)


def legend_markers(ax, extra=()):
    h, lab = ax.get_legend_handles_labels()
    h += [Line2D([], [], color="k", lw=1.2, label="model"),
          Line2D([], [], marker="o", color="k", ls="", label="measured (fit)"),
          Line2D([], [], marker="o", mfc="none", color="k", ls="", label="measured (validation)"),
          *extra]
    ax.legend(handles=h, fontsize=7, ncol=2, loc="upper left")


def lines_vs_B(df, ycol, pred, ylabel, name, title, scale=1.0, cap=None):
    Ss = sorted(df.S.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(Ss)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for S, c in zip(Ss, colors):
        sub = df[(df.S == S) & (df.status == "ok")]
        ax.plot(BD, pred(S, BD) * scale, color=c, lw=1.2, label=f"S={S}")
        tr, va = sub[~sub.is_validation], sub[sub.is_validation]
        ax.scatter(tr.B, tr[ycol] * scale, color=c, s=22, zorder=3)
        ax.scatter(va.B, va[ycol] * scale, facecolors="none", edgecolors=c, s=45, lw=1.3, zorder=3)
    extra = []
    if cap is not None:
        ax.axhline(cap * scale, color="r", ls="--", lw=1)
        extra.append(Line2D([], [], color="r", ls="--", label="GPU memory"))
        oom = df[df.status != "ok"]
        if len(oom):
            ax.scatter(oom.B, eq.memory(oom.S.values, oom.B.values) * scale, marker="x", c="r", s=50, zorder=4)
            extra.append(Line2D([], [], marker="x", color="r", ls="", label="OOM (at predicted)"))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size B")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    legend_markers(ax, extra)
    save(fig, name)


def lines_vs_S(df, ycol, pred, ylabel, name, title, scale=1.0, Bs=(1, 8, 64, 256)):
    Sd = np.linspace(32, 512, 200)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(Bs)))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for B, c in zip(Bs, colors):
        sub = df[(df.B == B) & (df.status == "ok")]
        ax.plot(Sd, pred(Sd, B) * scale, color=c, lw=1.2, label=f"B={B}")
        tr, va = sub[~sub.is_validation], sub[sub.is_validation]
        ax.scatter(tr.S, tr[ycol] * scale, color=c, s=22, zorder=3)
        ax.scatter(va.S, va[ycol] * scale, facecolors="none", edgecolors=c, s=45, lw=1.3, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("image size S [px]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    legend_markers(ax)
    save(fig, name)


def parity(df, ycol, pred_vals, label, unit, name, scale=1.0):
    ok = df[df.status == "ok"]
    p = pred_vals(ok) * scale
    m = ok[ycol].values * scale
    fig, ax = plt.subplots(figsize=(6, 6))
    v = ok.is_validation.values
    sc = ax.scatter(m[~v], p[~v], c=np.log2(ok.B.values[~v]), cmap="viridis", s=20, label="fit")
    ax.scatter(m[v], p[v], facecolors="none", edgecolors="crimson", s=40, lw=1.2, label="validation")
    lo, hi = min(m.min(), p.min()) * 0.8, max(m.max(), p.max()) * 1.25
    xx = np.array([lo, hi])
    ax.plot(xx, xx, "k-", lw=1)
    ax.fill_between(xx, xx / 1.2, xx * 1.2, color="gray", alpha=0.15, label="±20%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"measured {label} [{unit}]")
    ax.set_ylabel(f"predicted {label} [{unit}]")
    ax.set_title(f"{label}: predicted vs measured")
    fig.colorbar(sc, ax=ax, label="log2 B")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, name)


def error_heatmap(df, ycol, pred_vals, label, name):
    ok = df[df.status == "ok"].copy()
    ok["err"] = 100 * (pred_vals(ok) - ok[ycol].values) / ok[ycol].values
    Ss, Bs = sorted(df.S.unique()), sorted(df.B.unique())
    M = np.full((len(Ss), len(Bs)), np.nan)
    for _, r in ok.iterrows():
        M[Ss.index(r.S), Bs.index(r.B)] = r.err
    lim = max(5.0, np.nanpercentile(np.abs(M), 95))
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
    for i in range(len(Ss)):
        for j in range(len(Bs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7)
            elif not ((df.S == Ss[i]) & (df.B == Bs[j]) & (df.status == "ok")).any():
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=6, color="gray")
    val_s = set(df[df.is_validation].S) - set(df[~df.is_validation].S)
    val_b = set(df[df.is_validation].B) - set(df[~df.is_validation].B)
    ax.set_xticks(range(len(Bs)), [f"{b}*" if b in val_b else str(b) for b in Bs])
    ax.set_yticks(range(len(Ss)), [f"{s}*" if s in val_s else str(s) for s in Ss])
    ax.set_xlabel("batch size B   (* = validation value)")
    ax.set_ylabel("image size S [px]")
    ax.set_title(f"{label}: relative error (pred − meas)/meas, %")
    fig.colorbar(im, ax=ax, label="%")
    save(fig, name)


def surface(df, ycol, pred, label, unit, name, scale=1.0):
    Sg, Bg = np.meshgrid(np.linspace(32, 512, 40), np.geomspace(1, 256, 40))
    Z = np.log10(pred(Sg, Bg) * scale)
    ok = df[df.status == "ok"]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(Sg, np.log2(Bg), Z, cmap="viridis", alpha=0.55, lw=0)
    v = ok.is_validation.values
    ax.scatter(ok.S[~v], np.log2(ok.B[~v]), np.log10(ok[ycol][~v] * scale), c="k", s=10, label="fit")
    ax.scatter(ok.S[v], np.log2(ok.B[v]), np.log10(ok[ycol][v] * scale), c="crimson", s=14, label="validation")
    ax.set_xlabel("S [px]")
    ax.set_ylabel("log2 B")
    ax.set_zlabel(f"log10 {label} [{unit}]")
    ax.set_title(f"{label}: model surface + measurements")
    ax.legend()
    save(fig, name)


def flops_plot(df):
    Sd = np.linspace(32, 512, 200)
    per = df.groupby("S").apply(lambda g: (g.flops_counted / g.B).median()).reset_index(name="f")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Sd, eq.flops(Sd, 1) / 1e9, "k-", label="FLOPs(S,1): all ops")
    ax.plot(Sd, eq.flops_matmul(Sd, 1) / 1e9, "b--", label="conv+linear only (2·MAC)")
    ax.scatter(per.S, per.f / 1e9, c="orange", s=40, zorder=3, label="torch FlopCounterMode")
    ax.set_xlabel("image size S [px]")
    ax.set_ylabel("GFLOPs per image (FLOPs / B)")
    ax.set_title("FLOPs: formula vs torch.utils.flop_counter  (FLOPs ∝ B exactly)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, "flops_vs_S.png")


def perf_vs_work(df, th):
    ok = df[df.status == "ok"]
    Ss = sorted(df.S.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(Ss)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for S, c in zip(Ss, colors):
        sub = ok[ok.S == S]
        F = eq.flops(S, BD)
        ax.plot(F, F / eq.latency(S, BD, th) / 1e12, color=c, lw=1)
        ax.scatter(sub.flops_pred, sub.flops_pred / sub.latency_s / 1e12, color=c, s=18, label=f"S={S}")
    ax.axhline(th["P"] / 1e12, color="r", ls="--", lw=1, label=f"fitted P = {th['P'] / 1e12:.2f} TFLOP/s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("work per forward, FLOPs(S,B)")
    ax.set_ylabel("achieved throughput [TFLOP/s]")
    ax.set_title("Throughput vs work: slope 1 = launch-bound, plateau = compute/memory-bound")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "throughput_vs_work.png")


def roofline(df, th):
    ok = df[df.status == "ok"]
    AI = ok.flops_pred / ok.bytes_pred
    perf = ok.flops_pred / ok.latency_s
    ai = np.geomspace(1e-1, 1e3, 200)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ai, np.minimum(th["P"], ai * th["BW"]) / 1e12, "k-", lw=1.5, label="fitted roofline")
    sc = ax.scatter(AI, perf / 1e12, c=np.log10(ok.flops_pred), cmap="viridis", s=20, label="whole network")
    for name, kind, f_s, f_c, q_s, q_c, q_w in eq.LAYERS:
        if q_s > 0 and f_s > 0 and kind in ("conv", "bn", "relu", "pool"):
            a = f_s / q_s
            ax.axvline(a, color="gray", lw=0.5, alpha=0.5)
            ax.text(a, 1.2 * th["P"] / 1e12, name, rotation=90, fontsize=6, va="bottom", ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity FLOPs/Bytes [FLOP/B]")
    ax.set_ylabel("achieved [TFLOP/s]")
    ax.set_title("Roofline (vertical lines: per-layer AI at large S·B)")
    fig.colorbar(sc, ax=ax, label="log10 FLOPs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "roofline.png")


def regime_map(df, th):
    Sg, Bg = np.meshgrid(np.linspace(32, 512, 160), np.geomspace(1, 256, 160))
    br = eq.latency_breakdown(Sg, Bg, th)
    tot = br["launch"] + br["memory"] + br["compute"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, key, cmap in zip(axs[:3], ("launch", "memory", "compute"), ("Blues", "Oranges", "Greens")):
        im = ax.pcolormesh(Sg, Bg, br[key] / tot, cmap=cmap, vmin=0, vmax=1, shading="auto")
        ax.contour(Sg, Bg, br[key] / tot, levels=[0.25, 0.5, 0.75], colors="k", linewidths=0.7)
        ax.scatter(df.S, df.B, c="k", s=4)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("S [px]")
        ax.set_ylabel("B")
        ax.set_title(f"{key}-bound share of predicted latency")
        fig.colorbar(im, ax=ax)
    ax = axs[3]
    Bline = np.geomspace(1, 256, 200)
    for S, ls in zip((32, 128, 224, 512), ("-", "--", "-.", ":")):
        b = eq.latency_breakdown(S, Bline, th)
        t = b["launch"] + b["memory"] + b["compute"]
        ax.plot(Bline, b["launch"] / t, color="#3182bd", ls=ls)
        ax.plot(Bline, b["memory"] / t, color="#e6550d", ls=ls)
        ax.plot(Bline, b["compute"] / t, color="#31a354", ls=ls, label=f"S={S}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("B")
    ax.set_ylabel("share")
    ax.set_title("blue launch / orange memory / green compute")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, "regime_map.png")


def layer_plots(th):
    path = os.path.join(RES, "layer_times.csv")
    if not os.path.exists(path):
        return
    import pandas as pd
    from calibrate import layer_table
    lt = layer_table(pd.read_csv(path))
    tl = eq._theta_latency(th)
    P_row = np.array([float(eq._layer_P(tl, n, b)) for n, b in zip(lt.layer, lt.B)])
    pred = eq._smax(tl["t_op"], lt.F / P_row, lt.Q / tl["BW"])
    kinds = ["conv", "bn", "relu", "pool", "gap", "fc"]
    colors = dict(zip(kinds, plt.cm.tab10(range(len(kinds)))))
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in kinds:
        m = lt.kind == k
        ax.scatter(lt.time_s[m] * 1e3, pred[m] * 1e3, s=8, color=colors[k], label=k, alpha=0.7)
    lo, hi = 1e-3, max(lt.time_s.max(), pred.max()) * 1.5e3
    ax.plot([lo, hi], [lo, hi], "k-", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured layer GPU time [ms] (CUDA events)")
    ax.set_ylabel("predicted smax(t_op, F/P, Q/BW) [ms]")
    ax.set_title("Per-layer: predicted (refined θ) vs measured")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "parity_layers.png")

    big = lt[lt.time_s > 100e-6]
    g = big.groupby("layer", sort=False).agg(F=("F", "sum"), Q=("Q", "sum"), t=("time_s", "sum"))
    order = [l[0] for l in eq.LAYERS if l[0] in g.index]
    g = g.loc[order]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].bar(g.index, g.F / g.t / 1e12, color="#31a354")
    axs[0].axhline(th["P"] / 1e12, color="k", ls="--", lw=1, label="fitted P")
    axs[0].set_ylabel("effective TFLOP/s")
    axs[1].bar(g.index, g.Q / g.t / 1e9, color="#e6550d")
    axs[1].axhline(th["BW"] / 1e9, color="k", ls="--", lw=1, label="fitted BW")
    axs[1].set_ylabel("effective GB/s")
    for ax in axs:
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Per-layer effective throughput on GPU-bound points (layer time > 100 µs)")
    save(fig, "per_layer_efficiency.png")

    # conv efficiency vs batch: shows the cuDNN algorithm switch
    convs = [l[0] for l in eq.LAYERS if l[1] == "conv"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    Ss = sorted(lt.S.unique())
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(Ss)))
    for ax, name in zip(axs.flat, convs):
        g = lt[lt.layer == name]
        for S, c in zip(Ss, colors):
            h = g[g.S == S].sort_values("B")
            ax.plot(h.B, h.F / h.time_s / 1e12, "o-", ms=3, lw=0.8, color=c, label=f"S={S}")
        Bd = np.geomspace(1, 256, 200)
        ax.plot(Bd, np.broadcast_to(eq._layer_P(tl, name, Bd), Bd.shape) / 1e12, "r--", lw=1.5, label="model P_l")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("batch size B")
    for ax in axs[:, 0]:
        ax.set_ylabel("achieved TFLOP/s (F_l / t_l)")
    axs[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Conv efficiency vs batch (CUDA events). Drop of conv1 at B > 64 = cuDNN heuristic switch")
    save(fig, "conv_efficiency_vs_B.png")


def clocks_plot(df):
    ok = df[df.status == "ok"].sort_values("flops_pred")
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(ok.flops_pred, ok.sm_clock_MHz, c=ok.power_W, cmap="inferno", s=22)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs per forward")
    ax.set_ylabel("SM clock during energy loop [MHz]")
    ax.set_title("T4 power cap: clock drops from 1590 MHz as power reaches 70 W")
    fig.colorbar(sc, ax=ax, label="average power [W]")
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "clock_vs_work.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load()
    theta = json.load(open(os.path.join(RES, "theta.json")))
    meta_p = os.path.join(RES, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    th, te = theta["latency"], theta["energy"]
    cap = meta.get("gpu_total_memory_bytes")

    lat = lambda S, B: eq.latency(S, B, th)  # noqa: E731
    en = lambda S, B: eq.energy(S, B, te)  # noqa: E731
    mem = lambda S, B: eq.memory(S, B)  # noqa: E731

    flops_plot(df)
    lines_vs_B(df, "latency_s", lat, "latency [ms]", "latency_vs_B.png", "Latency vs batch", 1e3)
    lines_vs_S(df, "latency_s", lat, "latency [ms]", "latency_vs_S.png", "Latency vs image size", 1e3)
    lines_vs_B(df, "memory_bytes", mem, "peak allocated memory [MiB]", "memory_vs_B.png",
               "Peak memory vs batch", 1 / 2**20, cap=cap)
    lines_vs_S(df, "memory_bytes", mem, "peak allocated memory [MiB]", "memory_vs_S.png",
               "Peak memory vs image size", 1 / 2**20)
    lines_vs_B(df, "energy_J", en, "energy per forward [mJ]", "energy_vs_B.png", "Energy vs batch", 1e3)
    lines_vs_S(df, "energy_J", en, "energy per forward [mJ]", "energy_vs_S.png", "Energy vs image size", 1e3)

    pv = {"latency_s": lambda d: eq.latency(d.S.values, d.B.values, th),
          "memory_bytes": lambda d: eq.memory(d.S.values, d.B.values),
          "energy_J": lambda d: eq.energy(d.S.values, d.B.values, te)}
    parity(df, "latency_s", pv["latency_s"], "latency", "ms", "parity_latency.png", 1e3)
    if theta.get("latency_simple"):
        ths = theta["latency_simple"]
        parity(df, "latency_s", lambda d: eq.latency(d.S.values, d.B.values, ths), "latency (simple, global P)",
               "ms", "parity_latency_simple.png", 1e3)
        error_heatmap(df, "latency_s", lambda d: eq.latency(d.S.values, d.B.values, ths),
                      "Latency (simple, global P)", "error_latency_simple.png")
    parity(df, "memory_bytes", pv["memory_bytes"], "memory", "MiB", "parity_memory.png", 1 / 2**20)
    parity(df, "energy_J", pv["energy_J"], "energy", "mJ", "parity_energy.png", 1e3)
    error_heatmap(df, "latency_s", pv["latency_s"], "Latency", "error_latency.png")
    error_heatmap(df, "memory_bytes", pv["memory_bytes"], "Memory", "error_memory.png")
    error_heatmap(df, "energy_J", pv["energy_J"], "Energy", "error_energy.png")
    surface(df, "latency_s", lat, "latency", "s", "surface_latency.png")
    surface(df, "memory_bytes", mem, "memory", "B", "surface_memory.png")
    surface(df, "energy_J", en, "energy", "J", "surface_energy.png")
    perf_vs_work(df, th)
    roofline(df, th)
    regime_map(df, th)
    layer_plots(th)
    if "sm_clock_MHz" in df:
        clocks_plot(df)


if __name__ == "__main__":
    main()
