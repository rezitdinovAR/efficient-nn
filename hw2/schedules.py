def _lin(a, b, t):
    return a + (b - a) * t


class Schedule:
    """(lr, momentum) на итерации i из T.

    const      - постоянный lr
    step       - lr * gamma^k после каждого milestone
    range      - LR range test, линейно lr_min -> lr_max
    triangular - один треугольник lr_min -> lr_max -> lr_min
    1cycle     - подъём pct_up, спуск pct_down, остаток - линейно до lr_min/final_div;
                 momentum идёт в противофазе
    """

    def __init__(self, kind, T, lr=0.1, lr_max=None, lr_min=None, pct_up=0.45, pct_down=0.45,
                 final_div=100.0, mom=0.9, mom_max=None, mom_min=None, milestones=(0.625, 0.875), gamma=0.1):
        self.kind, self.T = kind, T
        self.lr, self.lr_max, self.lr_min = lr, lr_max, lr_min
        self.pct_up, self.pct_down, self.final_div = pct_up, pct_down, final_div
        self.mom, self.mom_max, self.mom_min = mom, mom_max, mom_min
        self.milestones, self.gamma = milestones, gamma

    def __call__(self, i):
        T = self.T
        if self.kind == "const":
            return self.lr, self.mom
        if self.kind == "step":
            k = sum(i >= m * T for m in self.milestones)
            return self.lr * self.gamma ** k, self.mom
        if self.kind == "range":
            lo = self.lr_min or 0.0
            return _lin(lo, self.lr_max, i / max(T - 1, 1)), self.mom
        if self.kind in ("1cycle", "triangular"):
            up, down = int(self.pct_up * T), int(self.pct_down * T)
            if self.kind == "triangular":
                up = down = T // 2
            mmax = self.mom if self.mom_max is None else self.mom_max
            mmin = self.mom if self.mom_min is None else self.mom_min
            if i < up:
                t = i / max(up, 1)
                return _lin(self.lr_min, self.lr_max, t), _lin(mmax, mmin, t)
            if i < up + down:
                t = (i - up) / max(down, 1)
                return _lin(self.lr_max, self.lr_min, t), _lin(mmin, mmax, t)
            rest = T - up - down
            t = (i - up - down) / max(rest, 1)
            return _lin(self.lr_min, self.lr_min / self.final_div, t), mmax
        raise ValueError(self.kind)
