import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE = {"dark": "#1a4f9c", "mid": "#2a78d6", "light": "#8ab4ea"}     # 1cycle
ORANGE = {"dark": "#b4431a", "mid": "#eb6834", "light": "#f5ab8d"}   # step
AQUA, YELLOW, MAGENTA, RED = "#1baf7a", "#eda100", "#e87ba4", "#e34948"
TXT = "#52514e"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#9a9893", "xtick.color": TXT, "ytick.color": TXT,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
})


def _csv(p):
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None
    return df if len(df) else None


def load(name):
    df = _csv(os.path.join(RES, name, "log.csv"))
    if df is None:
        return None
    s = os.path.join(RES, name, "summary.json")
    df.attrs["summary"] = json.load(open(s)) if os.path.exists(s) else {}
    return df


def have(*names):
    return all(_csv(os.path.join(RES, n, "log.csv")) is not None for n in names)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def smooth(y, k=3):
    if k <= 1 or len(y) < k:
        return np.asarray(y)
    return pd.Series(y).rolling(k, center=True, min_periods=1).mean().to_numpy()


def fig_range():
    runs = [("range_bs128", "batch 128", BLUE["light"]), ("range_bs512", "batch 512", BLUE["mid"]),
            ("range_bs1000", "batch 1000", BLUE["dark"])]
    runs = [(n, l, c) for n, l, c in runs if have(n)]
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    for n, l, c in runs:
        df = load(n)
        ax[0].plot(df.lr, smooth(df.test_acc * 100), color=c, label=l)
    ax[0].set(xlabel="learning rate (растёт линейно)", ylabel="test accuracy, %", ylim=(0, 100),
              title="LR range test, ResNet-56")
    ax[0].legend()
    df = load("range_bs512")
    if df is not None:
        ax[1].plot(df.lr, smooth(df.train_loss), color=BLUE["mid"], label="train loss")
        ax[1].plot(df.lr, smooth(df.test_loss), color=ORANGE["mid"], label="test loss")
        ax[1].set(xlabel="learning rate", ylabel="cross-entropy", title="batch 512: train и test loss",
                  ylim=(0, max(3.0, float(np.nanmin(df.test_loss)) + 0.5)))
        ax[1].legend()
    save(fig, "range_test.png")


def fig_main():
    cyc = [("1cycle_e25", "1cycle, 25 эп.", BLUE["light"]), ("1cycle_e50", "1cycle, 50 эп.", BLUE["mid"]),
           ("1cycle_e100", "1cycle, 100 эп.", BLUE["dark"])]
    stp = [("step_e50", "step lr=0.35, 50 эп.", ORANGE["light"]), ("step_e100", "step lr=0.35, 100 эп.", ORANGE["mid"]),
           ("step_e200", "step lr=0.35, 200 эп.", ORANGE["dark"])]
    runs = [(n, l, c) for n, l, c in cyc + stp if have(n)]
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [3, 2]})
    for n, l, c in runs:
        df = load(n)
        ls = "-" if n.startswith("1cycle") else "--"
        ax[0].plot(df.epoch, df.test_acc * 100, color=c, ls=ls, label=l)
        ax[1].plot(df.epoch, df.lr, color=c, ls=ls)
    ax[0].set(xlabel="эпоха", ylabel="test accuracy, %", ylim=(30, 96), title="1cycle и piecewise-constant")
    ax[0].legend(fontsize=7.5, loc="lower right")
    ax[1].set(xlabel="эпоха", ylabel="learning rate", title="расписания LR")
    save(fig, "main_comparison.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for group, col, lab in [(cyc, BLUE["mid"], "1cycle (0.1 → 3 → 0.001)"), (stp, ORANGE["mid"], "step (0.35, ÷10 два раза)")]:
        pts = [(load(n).attrs["summary"].get("final_test_acc", np.nan) * 100, load(n).epoch.max()) for n, _, _ in group if have(n)]
        if pts:
            acc, ep = zip(*pts)
            ax.plot(ep, acc, "o-", color=col, label=lab)
            for e, a in zip(ep, acc):
                ax.annotate(f"{a:.1f}", (e, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color=TXT)
    ax.set(xlabel="длина обучения, эпох", ylabel="финальная test accuracy, %", title="точность от бюджета")
    ax.legend(fontsize=7.5)
    save(fig, "acc_vs_budget.png")


def fig_hf():
    runs = [("hf_const", "lr = 0.1 постоянный", ORANGE["mid"]), ("hf_clr", "CLR 0.1 → 3 (2000 ит.)", BLUE["mid"])]
    runs = [(n, l, c) for n, l, c in runs if _csv(os.path.join(RES, n, "hf_estimate.csv")) is not None]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for n, l, c in runs:
        df = _csv(os.path.join(RES, n, "hf_estimate.csv"))
        ax.plot(df["iter"], df.eps_star_ema, color=c, label=l)
    ax.set(xlabel="итерация", ylabel="оценка оптимального lr ε* (EMA, α=0.1)", title="оценка lr по формуле (8)")
    ax.legend()
    save(fig, "hf_estimate.png")


def fig_limited():
    cfg = [("sub10k", "10k примеров"), ("sub20k", "20k примеров")]
    cfg = [(p, l) for p, l in cfg if have(p + "_1cycle") or have(p + "_step")]
    if not cfg:
        return
    fig, ax = plt.subplots(1, len(cfg), figsize=(4.5 * len(cfg), 3.2), squeeze=False)
    for a, (p, l) in zip(ax[0], cfg):
        for suf, lab, c in [("_1cycle", "1cycle (1500 ит.)", BLUE["mid"]), ("_step", "step lr=0.35 (3000 ит.)", ORANGE["mid"])]:
            if have(p + suf):
                df = load(p + suf)
                a.plot(df["iter"], df.test_acc * 100, color=c, label=lab)
        a.set(xlabel="итерация", ylabel="test accuracy, %", title=f"ResNet-56, {l}", ylim=(30, 95))
        a.legend(fontsize=7.5, loc="lower right")
    save(fig, "limited_data.png")


def fig_ablation():
    wd = [("1cycle_e25_wd1e-3", "wd = 1e-3", RED), ("1cycle_e25", "wd = 1e-4", BLUE["mid"]), ("1cycle_e25_wd1e-5", "wd = 1e-5", AQUA)]
    lr = [("1cycle_e25_lr0.3", "lr 0.03 → 0.3", YELLOW), ("1cycle_e25_lr1", "lr 0.1 → 1", AQUA), ("1cycle_e25", "lr 0.1 → 3", BLUE["mid"]),
          ("adam_1cycle_e25", "Adam, lr 1e-4 → 3e-3", MAGENTA)]
    panels = [(wd, "weight decay, 1cycle 0.1 → 3"), (lr, "максимальный lr и оптимизатор, 25 эпох")]
    panels = [(r, t) for r, t in panels if sum(have(n) for n, _, _ in r) >= 2]
    if not panels:
        return
    fig, ax = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.2), squeeze=False)
    for a, (runs, title) in zip(ax[0], panels):
        for n, l, c in runs:
            if have(n):
                df = load(n)
                a.plot(df.epoch, df.test_acc * 100, color=c, label=l)
        a.set(xlabel="эпоха", ylabel="test accuracy, %", title=title, ylim=(10, 96))
        a.legend(fontsize=7.5, loc="lower right")
    save(fig, "ablation.png")


def fig_gap():
    runs = [("1cycle_e100", "1cycle 100 эп.", BLUE["mid"]), ("step_e100", "step 100 эп.", ORANGE["mid"])]
    runs = [(n, l, c) for n, l, c in runs if have(n)]
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for n, l, c in runs:
        df = load(n)
        ax[0].plot(df.epoch, df.train_loss, color=c, label=l + " train")
        ax[0].plot(df.epoch, df.test_loss, color=c, ls="--", label=l + " test")
        ax[1].plot(df.epoch, smooth((df.train_acc - df.test_acc) * 100, 5), color=c, label=l)
    ax[0].set(xlabel="эпоха", ylabel="cross-entropy", ylim=(0, 2), title="train / test loss")
    ax[0].legend(fontsize=7.5)
    ax[1].set(xlabel="эпоха", ylabel="train acc − test acc, п.п.", title="generalization gap")
    ax[1].legend(fontsize=7.5)
    save(fig, "generalization_gap.png")


def table():
    rows = []
    for s in sorted(glob.glob(os.path.join(RES, "*", "summary.json"))):
        m = json.load(open(s))
        rows.append(dict(run=m["name"], policy=m["policy"],
                         lr=(f'{m["lr_min"]}→{m["lr_max"]}' if m["policy"] in ("1cycle", "triangular", "range") else m["lr"]),
                         wd=m["wd"], bs=m["bs"], n_train=m["n_train"], iters=m["T"], epochs=round(m["T"] / m["iters_per_epoch"], 1),
                         final_acc=round(m["final_test_acc"] * 100, 2), best_acc=round(m["best_test_acc"] * 100, 2),
                         minutes=round(m["wall_time_s"] / 60, 1), device=m["device"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "summary.csv"), index=False)
    with open(os.path.join(RES, "summary.md"), "w") as f:
        f.write(df.to_markdown(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    fig_range(); fig_main(); fig_hf(); fig_limited(); fig_ablation(); fig_gap(); table()
