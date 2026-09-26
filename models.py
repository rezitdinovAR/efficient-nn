"""SmallCNN for HW1.

Sequential, no residuals. Every conv: padding=k//2, bias=False, followed by
BatchNorm2d and ReLU(inplace=True). eval(), FP32. Input 3 x S x S, 100 classes.

    Conv7x7 s2 3->32  + MaxPool3x3 s2 p1   -> S/4
    Conv5x5    32->64                       -> S/4
    Conv3x3 s2 64->128                      -> S/8
    Conv1x1    128->256                     -> S/8
    Conv3x3 s2 256->256                     -> S/16
    Conv1x1    256->512                     -> S/16
    GAP -> Linear 512->256 -> ReLU -> Linear 256->100

The network is a *flat* nn.Sequential on purpose: nn.Sequential.forward does
`x = layer(x)`, so the previous activation is freed right after the next one is
produced. This is exactly the liveness assumption used in equations.memory().
"""
import torch
import torch.nn as nn


def _cbr(cin, cout, k, stride=1, use_bn=True):
    layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)]
    if use_bn:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.ReLU(inplace=True))
    return layers


class SmallCNN(nn.Sequential):
    def __init__(self, num_classes: int = 100, use_bn: bool = True):
        super().__init__(
            *_cbr(3, 32, 7, 2, use_bn),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *_cbr(32, 64, 5, 1, use_bn),
            *_cbr(64, 128, 3, 2, use_bn),
            *_cbr(128, 256, 1, 1, use_bn),
            *_cbr(256, 256, 3, 2, use_bn),
            *_cbr(256, 512, 1, 1, use_bn),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )


def build_model(num_classes: int = 100, use_bn: bool = True) -> SmallCNN:
    return SmallCNN(num_classes=num_classes, use_bn=use_bn).eval()


if __name__ == "__main__":
    m = build_model()
    n_par = sum(p.numel() for p in m.parameters())
    n_buf = sum(b.numel() for b in m.buffers())
    print(m)
    print(f"parameters: {n_par:,}   buffers: {n_buf:,}")
    S = 224
    x = torch.randn(1, 3, S, S)
    with torch.inference_mode():
        for layer in m:
            x = layer(x)
            print(f"{layer.__class__.__name__:18s} -> {tuple(x.shape)}")
