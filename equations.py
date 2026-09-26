"""Closed-form cost model of SmallCNN (see hw1_handwritten.pdf for derivations).

All public functions accept Python scalars or NumPy arrays for `image_size` (S)
and `batch` (B) and broadcast them. Scalars in -> float out.

Units: FLOPs, bytes, seconds, joules.
"""
from __future__ import annotations

import numpy as np

FP32 = 4  # bytes per element

# ---------------------------------------------------------------------------
# Per-layer table (one entry per launched op, 23 ops).
#   FLOPs  F_l(S,B) = B * (f_s * S^2 + f_c)
#   Bytes  Q_l(S,B) = B * (q_s * S^2 + q_c) + q_w
# f_s, q_s : per image, per S^2      f_c, q_c : per image, constant
# q_w      : weights / BN params, read once per forward (independent of B)
# Q counts every tensor read/written once through DRAM (ideal caching), FP32;
# MaxPool also writes int64 indices (max_pool2d_with_indices on CUDA).
# ---------------------------------------------------------------------------
#          name       kind    f_s    f_c     q_s  q_c    q_w
LAYERS = [
    ("conv1",   "conv", 2352,      0,  44,    0,   18816),
    ("bn1",     "bn",     16,      0,  64,    0,     512),
    ("relu1",   "relu",    8,      0,  64,    0,       0),
    ("maxpool", "pool",   16,      0,  56,    0,       0),
    ("conv2",   "conv", 6400,      0,  24,    0,  204800),
    ("bn2",     "bn",      8,      0,  32,    0,    1024),
    ("relu2",   "relu",    4,      0,  32,    0,       0),
    ("conv3",   "conv", 2304,      0,  24,    0,  294912),
    ("bn3",     "bn",      4,      0,  16,    0,    2048),
    ("relu3",   "relu",    2,      0,  16,    0,       0),
    ("conv4",   "conv", 1024,      0,  24,    0,  131072),
    ("bn4",     "bn",      8,      0,  32,    0,    4096),
    ("relu4",   "relu",    4,      0,  32,    0,       0),
    ("conv5",   "conv", 4608,      0,  20,    0, 2359296),
    ("bn5",     "bn",      2,      0,   8,    0,    4096),
    ("relu5",   "relu",    1,      0,   8,    0,       0),
    ("conv6",   "conv", 1024,      0,  12,    0,  524288),
    ("bn6",     "bn",      4,      0,  16,    0,    8192),
    ("relu6",   "relu",    2,      0,  16,    0,       0),
    ("gap",     "gap",     2,    512,   8, 2048,       0),
    ("fc1",     "fc",      0, 262400,   0, 3072,  525312),
    ("relu_fc", "relu",    0,    256,   0, 2048,       0),
    ("fc2",     "fc",      0,  51300,   0, 1424,  102800),
]
N_OPS = len(LAYERS)  # 23

# Closed-form totals (sum of the table; asserted below)
F_S2, F_C = 17793, 314468            # FLOPs(S,B) = B (17793 S^2 + 314468)
F_MM_S2, F_MM_C = 17712, 313344      # conv+linear only (= torch FlopCounterMode)
Q_S2, Q_C, Q_W = 548, 8592, 4181264  # Q(S,B) = 4181264 + B (548 S^2 + 8592)

# ---------------------------------------------------------------------------
# Memory constants
# ---------------------------------------------------------------------------
N_PARAMS = 1_045_316                 # float32 params + BN running stats
# every parameter/buffer is a separate CUDA allocation rounded up to 512 B
PARAM_ALLOC_BYTES = 4_187_136
# Library workspaces allocated through the caching allocator on the first
# Linear call and kept alive (library constants, not fitted):
#   cuBLAS   : CUBLAS_WORKSPACE_CONFIG ":4096:2:16:8" = 2*4 MiB + 8*16 KiB (non-Hopper)
#   cuBLASLt : TORCH_CUBLASLT_WORKSPACE_SIZE default 1 MiB
# measured on T4 / torch 2.11: static - params = 9 568 256 B exactly.
CUBLAS_WS_BYTES = 2 * 4096 * 1024 + 8 * 16 * 1024 + 1024 * 1024  # 9_568_256
# live activations at the peak (BatchNorm-1 step): input x (3 S^2) +
# conv1 output (8 S^2) + bn1 output (8 S^2), FP32
ACT_PEAK_S2 = FP32 * (3 + 8 + 8)     # 76 bytes per image per S^2

K_SMOOTH = 6.0  # smooth max: (a^k + b^k + c^k)^(1/k)

assert sum(l[2] for l in LAYERS) == F_S2 and sum(l[3] for l in LAYERS) == F_C
assert sum(l[4] for l in LAYERS) == Q_S2 and sum(l[5] for l in LAYERS) == Q_C
assert sum(l[6] for l in LAYERS) == Q_W == FP32 * N_PARAMS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sb(image_size, batch):
    S, B = np.broadcast_arrays(np.asarray(image_size, dtype=np.float64),
                               np.asarray(batch, dtype=np.float64))
    return S, B


def _out(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def _round512(nbytes):
    return np.maximum(512.0, np.ceil(np.asarray(nbytes, dtype=np.float64) / 512.0) * 512.0)


def _smax(*terms, k=K_SMOOTH):
    terms = np.broadcast_arrays(*[np.asarray(t, dtype=np.float64) for t in terms])
    m = np.maximum.reduce(terms)
    m_safe = np.where(m > 0, m, 1.0)
    s = sum((t / m_safe) ** k for t in terms)
    return np.where(m > 0, m_safe * s ** (1.0 / k), 0.0)


def layer_terms(image_size, batch):
    """Yield (name, kind, F_l, Q_l) arrays for every op."""
    S, B = _sb(image_size, batch)
    S2 = S * S
    for name, kind, f_s, f_c, q_s, q_c, q_w in LAYERS:
        yield name, kind, B * (f_s * S2 + f_c), B * (q_s * S2 + q_c) + q_w


def _theta_latency(theta):
    """theta: (t0, t_op, P, BW) or dict. Optional dict keys for per-layer efficiency:
    P_layer {name: FLOP/s}, P_layer_bigB {name: FLOP/s} used when B > B_switch."""
    if isinstance(theta, dict):
        th = dict(theta.get("latency", theta))
    else:
        t0, t_op, P, BW = theta
        th = {"t0": t0, "t_op": t_op, "P": P, "BW": BW}
    th.setdefault("P_layer", {})
    th.setdefault("P_layer_bigB", {})
    th.setdefault("B_switch", np.inf)
    return th


def _layer_P(th, name, B):
    P = th["P_layer"].get(name, th["P"])
    if name in th["P_layer_bigB"]:
        P = np.where(B > th["B_switch"], th["P_layer_bigB"][name], P)
    return P


# ---------------------------------------------------------------------------
# 1. FLOPs
# ---------------------------------------------------------------------------
def flops(image_size, batch):
    """All FLOPs of one forward pass (conv/linear 2*MAC + bias + BN + ReLU + pool + GAP)."""
    S, B = _sb(image_size, batch)
    return _out(B * (F_S2 * S * S + F_C))


def flops_matmul(image_size, batch):
    """Conv + Linear only, 2 FLOPs per MAC, no bias: what FlopCounterMode reports."""
    S, B = _sb(image_size, batch)
    return _out(B * (F_MM_S2 * S * S + F_MM_C))


# ---------------------------------------------------------------------------
# 2. Memory (peak torch.cuda.max_memory_allocated during one forward)
# ---------------------------------------------------------------------------
def memory(image_size, batch, lib_overhead=CUBLAS_WS_BYTES):
    """Peak allocated bytes = params + library workspace + live activations at the peak."""
    S, B = _sb(image_size, batch)
    S2 = S * S
    x_in = _round512(FP32 * 3 * S2 * B)
    conv1 = _round512(FP32 * 8 * S2 * B)
    bn1 = _round512(FP32 * 8 * S2 * B)
    return _out(PARAM_ALLOC_BYTES + lib_overhead + x_in + conv1 + bn1)


# ---------------------------------------------------------------------------
# 3. Bytes moved (DRAM traffic, ideal)
# ---------------------------------------------------------------------------
def bytes_moved(image_size, batch):
    S, B = _sb(image_size, batch)
    return _out(Q_W + B * (Q_S2 * S * S + Q_C))


# ---------------------------------------------------------------------------
# 4. Latency
#   T = t0 + sum_l smax(t_op, F_l / P_l, Q_l / BW)
#   simple : P_l = P for every layer, theta = (t0, t_op, P, BW)
#   refined: P_l per conv layer; conv1 has a second value for B > B_switch
#            (cuDNN heuristic switches the conv1 algorithm, see README)
# ---------------------------------------------------------------------------
def latency(image_size, batch, theta):
    th = _theta_latency(theta)
    _, B = _sb(image_size, batch)
    total = th["t0"]
    for name, _, F, Q in layer_terms(image_size, batch):
        total = total + _smax(th["t_op"], F / _layer_P(th, name, B), Q / th["BW"])
    return _out(total)


def latency_breakdown(image_size, batch, theta):
    """Time attributed to the dominant term of every op: launch / memory / compute."""
    th = _theta_latency(theta)
    S, B = _sb(image_size, batch)
    out = {"launch": np.full(S.shape, float(th["t0"])), "memory": np.zeros(S.shape),
           "compute": np.zeros(S.shape)}
    for name, _, F, Q in layer_terms(image_size, batch):
        tl = np.broadcast_to(np.float64(th["t_op"]), F.shape)
        tc, tm = F / _layer_P(th, name, B), Q / th["BW"]
        t = _smax(tl, tc, tm)
        dom = np.argmax(np.stack([tl, tm, tc]), axis=0)
        out["launch"] += np.where(dom == 0, t, 0.0)
        out["memory"] += np.where(dom == 1, t, 0.0)
        out["compute"] += np.where(dom == 2, t, 0.0)
    return {k: _out(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 5. Energy
#   E = P_static * T(S,B,theta) + e_flop * FLOPs + e_byte * Bytes
#   theta_energy = {"P_static", "e_flop", "e_byte", "latency": {...}}
#   (the full theta.json dict is accepted too)
# ---------------------------------------------------------------------------
def energy(image_size, batch, theta_energy):
    te = theta_energy
    if "energy" in te:                       # full theta.json
        lat, te = te["latency"], te["energy"]
    else:
        lat = te["latency"]
    T = latency(image_size, batch, lat)
    return _out(te["P_static"] * np.asarray(T)
                + te["e_flop"] * np.asarray(flops(image_size, batch))
                + te["e_byte"] * np.asarray(bytes_moved(image_size, batch)))


if __name__ == "__main__":
    th = {"t0": 30e-6, "t_op": 10e-6, "P": 4.5e12, "BW": 250e9}
    for S, B in [(32, 1), (224, 1), (224, 64), (512, 256)]:
        print(f"S={S:3d} B={B:3d}  FLOPs={flops(S, B):.3e}  Mem={memory(S, B) / 2**20:9.1f} MiB  "
              f"Q={bytes_moved(S, B) / 2**20:9.1f} MiB  T={latency(S, B, th) * 1e3:8.3f} ms")
    Sg, Bg = np.meshgrid([32, 64, 128], [1, 2, 4])
    print("broadcast ok:", flops(Sg, Bg).shape, latency(Sg, Bg, th).shape)
