import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, bn_momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.downsample = stride != 1 or cin != cout
        self.pad = cout - cin

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            sc = F.avg_pool2d(x, 2)
            sc = F.pad(sc, (0, 0, 0, 0, 0, self.pad))
        else:
            sc = x
        return F.relu(out + sc)


class CifarResNet(nn.Module):
    def __init__(self, depth=56, num_classes=10, bn_momentum=0.05, dropout=0.0):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16, momentum=bn_momentum)
        layers, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for i in range(n):
                layers.append(BasicBlock(cin, cout, stride if i == 0 else 1, bn_momentum))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(self.dropout(x))


def build_model(depth=56, **kw):
    return CifarResNet(depth, **kw)


if __name__ == "__main__":
    for d in (20, 56, 110):
        print(d, sum(p.numel() for p in build_model(d).parameters()) / 1e6, "M")
