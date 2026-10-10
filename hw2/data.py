import os
import numpy as np
import torch


MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar10(root, device, n_train=None, seed=0):
    npz = os.path.join(root, "cifar10.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} не найден, сначала python prepare_data.py")
    d = np.load(npz)
    xtr = torch.from_numpy(d["xtr"]).permute(0, 3, 1, 2).contiguous()
    ytr = torch.from_numpy(d["ytr"]).long()
    xte = torch.from_numpy(d["xte"]).permute(0, 3, 1, 2).contiguous()
    yte = torch.from_numpy(d["yte"]).long()
    if n_train is not None and n_train < len(xtr):
        # подвыборка с одинаковым числом примеров на класс
        g = np.random.default_rng(seed)
        idx = np.concatenate([g.permutation(np.where(ytr.numpy() == c)[0])[: n_train // 10] for c in range(10)])
        idx = torch.from_numpy(np.sort(idx))
        xtr, ytr = xtr[idx], ytr[idx]
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)


def normalize(x_uint8):
    x = x_uint8.float().div_(255.0)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class TrainLoader:
    # бесконечный итератор по батчам; random crop (pad 4) + flip

    def __init__(self, x, y, batch_size, augment=True, seed=0):
        self.x, self.y, self.bs, self.augment = x, y, batch_size, augment
        self.n = len(x)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.xpad = torch.nn.functional.pad(x, (4, 4, 4, 4)) if augment else x
        self.ar = torch.arange(32, device=x.device)
        self.iters_per_epoch = self.n // batch_size

    def __iter__(self):
        while True:
            perm = torch.randperm(self.n, generator=self.gen).to(self.x.device)
            for i in range(self.iters_per_epoch):
                yield self._batch(perm[i * self.bs:(i + 1) * self.bs])

    def _batch(self, idx):
        b = len(idx)
        if not self.augment:
            return normalize(self.x[idx]), self.y[idx]
        dy = torch.randint(0, 9, (b,), generator=self.gen).to(self.x.device)
        dx = torch.randint(0, 9, (b,), generator=self.gen).to(self.x.device)
        flip = (torch.rand(b, generator=self.gen) < 0.5).to(self.x.device)
        rows = dy[:, None] + self.ar[None, :]
        cols = dx[:, None] + self.ar[None, :]
        cols = torch.where(flip[:, None], cols.flip(1), cols)
        xb = self.xpad[idx]
        bi = torch.arange(b, device=xb.device)[:, None, None]
        out = xb[bi, :, rows[:, :, None], cols[:, None, :]]   # b,32,32,3
        out = out.permute(0, 3, 1, 2).contiguous()
        return normalize(out), self.y[idx]


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(x), batch_size):
        xb, yb = normalize(x[i:i + batch_size]), y[i:i + batch_size]
        out = model(xb)
        loss_sum += torch.nn.functional.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    model.train()
    return loss_sum / len(x), correct / len(x)
