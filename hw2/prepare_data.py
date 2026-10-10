# собирает data/cifar10.npz: с HF (parquet, быстро) или через torchvision с cs.toronto.edu (python prepare_data.py torchvision)
import io
import os
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torchvision
from PIL import Image

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NPZ = os.path.join(DATA, "cifar10.npz")
HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def from_torchvision():
    tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
    te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def from_hf():
    out = []
    for split in ("train", "test"):
        path = os.path.join(DATA, f"{split}.parquet")
        if not os.path.exists(path):
            print("downloading", HF.format(split))
            urllib.request.urlretrieve(HF.format(split), path)
        t = pq.read_table(path).to_pydict()
        x = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        y = np.array(t["label"], dtype=np.int64)
        out += [x, y]
    return out


def main():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(NPZ):
        print("exists:", NPZ)
        return
    src = sys.argv[1] if len(sys.argv) > 1 else "hf"
    xtr, ytr, xte, yte = from_hf() if src == "hf" else from_torchvision()
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    assert np.bincount(ytr).tolist() == [5000] * 10
    np.savez(NPZ, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    print("saved", NPZ, xtr.shape, xte.shape)


if __name__ == "__main__":
    main()
