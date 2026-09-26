# HW1 — Analytical performance model of a small CNN

## Setup

| | |
|---|---|
| GPU | Tesla T4 (Google Colab), CC 7.5, 40 SMs, 15.6 GB, power limit 70 W, idle 26.4 W |
| Software | PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0, driver 580.82.07, Python 3.13.15 |
| Energy | NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`), ≥ 3 s window per config |
| Flags | `cudnn.benchmark=False`, TF32 off, `eval()` + `inference_mode()`, FP32 |

## Reproduce

Colab (T4): open `hw1_colab.ipynb` and run all cells, or:

```bash
pip install nvidia-ml-py
python measure.py      # ~20 min -> results/measurements.csv, layer_times.csv, meta.json
python calibrate.py    # -> results/theta.json
python plots.py        # -> results/figures/*.png
```

Grid: S ∈ {32, 64, 128, 224, 256, 384, 512} ∪ {48, 112, 352, 400}, B ∈ {1, 2, …, 256} ∪ {95, 98, 167} (random, seed 2026) → 132 configs. θ is fitted on the base grid only (63 points); every config with a random S or B (69 points) is validation. Configs are measured in shuffled order to avoid thermal bias.

Per config: 5 warm-up passes, peak memory of one pass, latency = median of ≥ 10 synchronized passes (≥ 1 s), energy = NVML counter over ≥ 3 s of back-to-back passes divided by the number of passes, per-layer GPU time from CUDA events (`layer_times.csv`), FLOPs from `torch.utils.flop_counter` on the meta device.

## Model (derivation: `hw1_handwritten.pdf`)

Conv–BN–ReLU blocks (BN after every conv, since convs have no bias).

| Function | Formula |
|---|---|
| FLOPs | `B (17793 S² + 314468)`; conv+linear only `B (17712 S² + 313344)` |
| Memory | `4 187 136 + 9 568 256 + 76 B S²` bytes (params + cuBLAS/cuBLASLt workspaces + input, conv1 and bn1 outputs alive at the BN1 peak) |
| Bytes moved | `4 181 264 + B (548 S² + 8592)` bytes |
| Latency | `t0 + Σ_{23 ops} smax(t_op, F_l / P_l, Q_l / BW)`, smax = 6-norm |
| Energy | `P_static · T + e_flop · FLOPs + e_byte · Q` |

Latency is fitted in two steps, because P and BW cannot be separated from end-to-end time (both terms scale as B·S²): P_l and BW from per-layer CUDA-event times of GPU-bound layers (> 100 µs), then t0 and t_op from end-to-end latency. Two variants are reported: *simple* (one P for all convs) and *refined* (P_l per conv, conv1 with a separate value for B > 64).

## Results

Fitted θ (`results/theta.json`):

| | t0 | t_op | BW | P |
|---|---|---|---|---|
| simple | ≈ 0 µs | 34.3 µs | 231 GB/s | 2.88 TFLOP/s |
| refined | 144 µs | 30.0 µs | 231 GB/s | conv1 2.85 (B ≤ 64) / 0.65 (B > 64), conv2 6.27, conv3 4.13, conv4 3.95, conv5 4.10, conv6 3.71 TFLOP/s |

Energy: P_static = 47.0 W, e_flop = 0 pJ/FLOP, e_byte = 257 pJ/B.

Errors (APE, %; fit = base grid, val = unseen points):

| | FLOPs vs counter | Memory | Latency simple | Latency refined | Energy | Energy with measured T |
|---|---|---|---|---|---|---|
| MAPE fit / val | 0 / 0 | 14.7 / 20.3 | 22.8 / 18.8 | 8.6 / 9.1 | 8.5 / 12.1 | 8.1 / 8.0 |
| median fit / val | 0 / 0 | 14.8 / 19.5 | 23.9 / 14.5 | 5.6 / 5.5 | 4.8 / 8.4 | 5.7 / 6.5 |
| max fit / val | 0 / 0 | 68 / 63 | 53 / 48 | 47 / 42 | 52 / 47 | 50 / 22 |

OOM: measured 0, predicted 0 (132/132 agree). Largest config S = 512, B = 256: predicted 5.11 GB, measured 6.17 GB, device 15.6 GB.

Main figures: `latency_vs_B.png`, `memory_vs_B.png`, `energy_vs_B.png` (and `*_vs_S`), `parity_*`, `error_*` (relative-error heatmaps over the grid), `surface_*`, `regime_map.png`, `throughput_vs_work.png`, `roofline.png`, `conv_efficiency_vs_B.png`, `per_layer_efficiency.png`, `parity_layers.png`, `clock_vs_work.png`.

## Discussion

**FLOPs** are exact: `FlopCounterMode` matches the conv+linear formula to 0% at all 132 points. Elementwise ops (BN, ReLU, pooling) add only 0.46% of FLOPs but 73% of the bytes moved (400 of 548 B per S² per image). This is why FLOPs alone do not predict time.

**Memory.** The formula is a strict lower bound: the measurement is never below it, and it is exact (< 0.5%) at 29 of 132 points. These are the points where cuDNN picks algorithms without workspace, mostly S ≥ 224 with B ≤ 8. The peak is at the first BatchNorm: the input, the conv1 output and the BN1 output are alive at once, 19 S² floats per image. The library constant was checked directly: allocated memory after warm-up minus parameters is 9 568 256 B, which is exactly cuBLAS (8.125 MiB) + cuBLASLt (1 MiB). All excess is cuDNN workspace chosen by the heuristic (`benchmark=False`). It shows up as steps in `error_memory.png`. At small B·S² it is a fixed ~5–10 MB; above a batch threshold (B ≥ 16 for S ≥ 256, B ≥ 64 for S = 128, B ≥ 95 for S ≤ 112) it grows ∝ B·S², reaching +20% (+1 GB) at S = 512, B = 256. The workspace depends on which algorithm is picked, which is not knowable in closed form. The OOM prediction is still right: even +20% is far below 15.6 GB, and no config on the grid ran out of memory.

**Latency — regimes** (`regime_map.png`, `throughput_vs_work.png`):
1. *Launch-bound.* For B·S² ≲ 1.4·10⁵ (S = 224: B ≲ 3; S = 64: B ≲ 35), latency is flat at t0 + 23·t_op ≈ 0.83 ms, and throughput grows linearly with work. The GPU waits for the PyTorch dispatcher on Colab's CPU (≈ 30 µs per op).
2. *Memory-bound.* BN, ReLU, MaxPool and GAP have AI ≤ 0.3 FLOP/B and always run at 218–242 GB/s (≈ 72% of the T4's 320 GB/s). At large sizes they take ≈ 20–30% of the time, although they are 0.5% of FLOPs.
3. *Compute-bound.* At large B·S² latency is ∝ B·S², and the convs take ≈ 70–80% of the time at 3.7–6.3 TFLOP/s (at throttled clocks, see below).

**Where the latency model breaks:**
- *cuDNN algorithm switch.* conv1 (C_in = 3, 7×7) runs at 2.85 TFLOP/s for B ≤ 64 and at 0.65 TFLOP/s for B > 64, at every S (`conv_efficiency_vs_B.png`). The heuristic switches the conv1 algorithm on batch size. With one global P this appears as a systematic +30% error for B ≤ 64 and −10…−50% for B > 64 (`error_latency_simple.png`). This single effect accounts for most of the simple model's 20% MAPE. The refined model with per-layer P and a batch threshold cuts the error to 9%. The threshold lies somewhere in (64, 95], since the grid has no points in between.
- *Small spatial maps.* At S = 32–48, conv5/conv6 outputs are 2×2 or 3×3. That is too few thread blocks for 40 SMs, so efficiency stays at 0.3–0.6 TFLOP/s. The model underpredicts S ≤ 112 at large B by 15–45%. Efficiency also ramps up gradually with B (occupancy), while the smax only switches between asymptotes.
- *Power cap / clocks* (`clock_vs_work.png`). The board draws 46–70 W, and under load the SM clock drops from 1590 MHz to 690–900 MHz. P_l is fitted at throttled clocks, so light GPU-bound configs (S ≥ 256, B = 1–2) run faster than predicted (+20%).
- *Identifiability.* Only t0 + 23·t_op is well determined from the flat launch-bound region; in the simple fit t0 collapses to 0.

**Energy.** E ≈ 47 W·T + 0.26 nJ/B·Q, fitted with non-negative least squares on relative error. e_flop → 0 because FLOPs and bytes are almost collinear (both ∝ B·S²), and because power is nearly constant: 60–70 W for almost every config, pinned by the 70 W cap. So energy is essentially P̄·T. P_static = 47 W is well above the 26 W idle: it is the board's power with clocks up, not true idle. With measured T the model is at 8% MAPE; with predicted T it inherits the latency errors (8.5 / 12%). Consequences:
- Energy per image at S = 224 drops from 52 mJ (B = 1) to 21 mJ (B = 64), because amortizing the launch-bound floor is the main energy lever.
- It then jumps back to ≈ 30 mJ for B ≥ 95 because of the conv1 algorithm switch. On this GPU and software stack, B = 64 is the most energy-efficient batch; a larger batch is worse.
