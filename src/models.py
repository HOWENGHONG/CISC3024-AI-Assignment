"""
Model definitions for AI Assignment #1 (CISC3024 Pattern Recognition).

Main algorithm
--------------
ConvNeXt  --  Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie,
              "A ConvNet for the 2020s", CVPR 2022.  arXiv:2201.03545

The original ConvNeXt is designed for 224x224 ImageNet images.  The version
implemented here keeps every architectural ingredient of the paper (patchify
stem, 4-stage hierarchical design with stage ratio 1:1:3:1, depthwise 7x7
convolution as the "token mixer", inverted bottleneck MLP with expansion 4,
LayerNorm instead of BatchNorm, GELU instead of ReLU, a single normalisation
and a single activation per block, separate downsampling layers, layer scale
and stochastic depth) but adapts the stem and the stage resolutions to the
32x32 images of CIFAR-10.

For the ablation study, the block can be switched back to the "ResNet style"
micro design (3x3 kernel, BatchNorm + ReLU), which lets us re-trace part of the
"modernisation roadmap" of the paper on a small dataset.  A standard CIFAR
ResNet-18 is also provided as a classic deep-CNN baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor.

    ConvNeXt uses a *single* LayerNorm per block (applied over the channel
    axis for every spatial position), replacing the BatchNorm layers of the
    ResNet block.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> normalise over C
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic depth (Huang et al., 2016) applied per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """One ConvNeXt block (Fig. 4 of the paper).

        x ─► DWConv k×k ─► LN ─► Linear (C→4C) ─► GELU ─► Linear (4C→C)
          │                                                    │
          │                                       × layer_scale γ
          └────────────────────── + ◄──────── DropPath ◄───────┘

    The two 1x1 convolutions are implemented as nn.Linear on a channels-last
    tensor, exactly as in the official implementation.

    Parameters
    ----------
    dim          : number of channels C
    kernel_size  : depthwise kernel size (7 in the paper, 3 for the ablation)
    mlp_ratio    : expansion factor of the inverted bottleneck (4 in the paper)
    drop_path    : stochastic depth rate of this block
    layer_scale  : initial value of the per-channel scale γ (1e-6 in the paper),
                   set to <= 0 to disable
    norm         : "ln" (ConvNeXt) or "bn" (ResNet-style ablation)
    act          : "gelu" (ConvNeXt) or "relu" (ResNet-style ablation)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
        layer_scale: float = 1e-6,
        norm: str = "ln",
        act: str = "gelu",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=dim
        )
        if norm == "ln":
            self.norm = nn.LayerNorm(dim, eps=1e-6)  # applied channels-last
            self.channels_last_norm = True
        elif norm == "bn":
            self.norm = nn.BatchNorm2d(dim)
            self.channels_last_norm = False
        else:
            raise ValueError(norm)

        hidden = mlp_ratio * dim
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale > 0 else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        if not self.channels_last_norm:
            x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if self.channels_last_norm:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        return shortcut + self.drop_path(x)


# --------------------------------------------------------------------------- #
# ConvNeXt for CIFAR-10
# --------------------------------------------------------------------------- #
class ConvNeXtCIFAR(nn.Module):
    """A small ConvNeXt adapted to 32x32 inputs.

    Stage resolutions: 32 -> 16 -> 8 -> 4 (stem keeps full resolution, then
    three 2x2/stride-2 "separate downsampling layers", each preceded by LN).

    Default configuration ("ConvNeXt-CIFAR", sized so that it can be trained
    on a CPU in a reasonable time - see the report):
        depths = (1, 1, 3, 1)      # stage compute ratio 1:1:3:1, as in the paper
        dims   = (32, 64, 128, 256)
        -> 1.18 M parameters
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        depths=(1, 1, 3, 1),
        dims=(32, 64, 128, 256),
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.1,
        layer_scale: float = 1e-6,
        norm: str = "ln",
        act: str = "gelu",
    ):
        super().__init__()
        make_norm = (lambda c: LayerNorm2d(c)) if norm == "ln" else (lambda c: nn.BatchNorm2d(c))

        # Stem.  ImageNet ConvNeXt uses a 4x4/stride-4 "patchify" convolution.
        # On 32x32 images that would leave only 8x8 positions, so we use a
        # stride-1 3x3 convolution followed by the norm layer instead.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            make_norm(dims[0]),
        )

        # Separate downsampling layers: LN -> Conv 2x2 stride 2
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    make_norm(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        # Stages with linearly increasing stochastic-depth rate
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = [
                ConvNeXtBlock(
                    dims[i],
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    drop_path=dp_rates[cur + j],
                    layer_scale=layer_scale,
                    norm=norm,
                    act=act,
                )
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        # Head: global average pooling -> LN -> Linear
        self.head_norm = nn.LayerNorm(dims[-1], eps=1e-6) if norm == "ln" else nn.BatchNorm1d(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages[0](x)
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i + 1](x)
        return x.mean([-2, -1])  # global average pooling -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head_norm(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Baseline: the classic CIFAR ResNet of He et al. (CVPR 2016, Sec. 4.2)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    """ResNet basic block: 3x3 conv-BN-ReLU-3x3 conv-BN + identity, then ReLU."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CIFARResNet(nn.Module):
    """ResNet-(6n+2) for CIFAR: 3 stages at 32x32, 16x16 and 8x8 resolution.

    n=3 and widths=(16, 32, 64) gives the original ResNet-20 (0.27 M params);
    widths=(32, 64, 128) doubles every width ("ResNet-20 x2", 1.08 M params),
    which matches the parameter count of ConvNeXt-CIFAR.
    """

    def __init__(self, n: int = 3, widths=(32, 64, 128), num_classes: int = 10):
        super().__init__()
        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(widths[0], n, 1)
        self.layer2 = self._make_layer(widths[1], n, 2)
        self.layer3 = self._make_layer(widths[2], n, 2)
        self.fc = nn.Linear(widths[2], num_classes)

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
MODEL_CONFIGS = {
    # main algorithm
    "convnext": dict(kernel_size=7, norm="ln", act="gelu"),
    # ablation 1: shrink the depthwise kernel back to 3x3
    "convnext_k3": dict(kernel_size=3, norm="ln", act="gelu"),
    # ablation 2: revert the micro design to BatchNorm + ReLU
    "convnext_bnrelu": dict(kernel_size=7, norm="bn", act="relu"),
    # ablation 3: disable layer scale and stochastic depth
    "convnext_noreg": dict(kernel_size=7, norm="ln", act="gelu", layer_scale=0.0, drop_path_rate=0.0),
}
BASELINES = {
    "resnet20": dict(n=3, widths=(16, 32, 64)),
    "resnet20x2": dict(n=3, widths=(32, 64, 128)),
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name in BASELINES:
        return CIFARResNet(num_classes=num_classes, **BASELINES[name])
    if name in MODEL_CONFIGS:
        return ConvNeXtCIFAR(num_classes=num_classes, **MODEL_CONFIGS[name])
    raise ValueError(f"unknown model '{name}'. Choose from {list(MODEL_CONFIGS) + list(BASELINES)}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_size=(1, 3, 32, 32)) -> int:
    """Multiply-accumulate operations of one forward pass (FLOPs / 2)."""
    from torch.utils.flop_counter import FlopCounterMode

    model = model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.zeros(*input_size))
    return fc.get_total_flops() // 2


if __name__ == "__main__":
    # quick self-check: forward pass, parameter and MAC counts of every model
    x = torch.randn(2, 3, 32, 32)
    for name in list(MODEL_CONFIGS) + list(BASELINES):
        m = build_model(name)
        y = m(x)
        print(f"{name:18s} params = {count_parameters(m)/1e6:6.2f} M   "
              f"MACs = {count_macs(m)/1e6:7.1f} M   output = {tuple(y.shape)}")
