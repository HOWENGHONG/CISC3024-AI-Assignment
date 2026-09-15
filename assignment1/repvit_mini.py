"""
RepViT-mini: a small, faithful reproduction of the core ideas in

    Wang, Chen, Lin, Han, Ding. "RepViT: Revisiting Mobile CNN From ViT
    Perspective." CVPR 2024. (arXiv:2307.09283)

adapted to run on CPU, on a tiny grayscale digit-recognition dataset
(scikit-learn's `load_digits`, 8x8 images, 10 classes) instead of
ImageNet, because this sandbox has no GPU and no access to image
dataset mirrors.

Core ideas reproduced from the paper:
  1. "Separate the token mixer and the channel mixer" (MetaFormer-style
     block), instead of MobileNetV3's inverted-residual block that mixes
     spatial and channel information together.
  2. Token mixer = a depthwise convolution trained with *structural
     re-parameterization* (RepVGG-style multi-branch training,
     single-branch inference): during training it is
         BN(DWConv3x3(x)) + BN(DWConv1x1(x)) + BN(x)
     and after training the three branches are mathematically fused
     into ONE depthwise 3x3 convolution with a bias term, which is
     exactly equivalent (verified numerically below) but faster and
     smaller at inference time.
  3. Channel mixer = a two-layer pointwise ("1x1 conv") feed-forward
     block with an expansion ratio, i.e. the ViT-FFN written as
     convolutions, with a residual connection.
  4. Squeeze-and-Excitation (SE) channel attention inserted after every
     other token mixer, as in the paper.
  5. All normalisation is BatchNorm (never LayerNorm), specifically
     because BN can be algebraically folded into the preceding
     convolution at inference time -- which is what step 2 relies on.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------------
# 1. RepVGGDW: the reparameterizable depthwise "token mixer"
# --------------------------------------------------------------------------
class RepVGGDW(nn.Module):
    """Multi-branch depthwise conv (train time) <-> single depthwise
    conv (inference time), after Ding et al. RepVGG (CVPR 2021), used by
    RepViT as its token mixer."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.bn3x3 = nn.BatchNorm2d(dim)
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
        self.bn1x1 = nn.BatchNorm2d(dim)
        self.bn_id = nn.BatchNorm2d(dim)
        self.fused = None  # filled in by self.fuse()

    def forward(self, x):
        if self.fused is not None:
            return self.fused(x)
        return self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x)) + self.bn_id(x)

    @staticmethod
    def _fuse_conv_bn(kernel, bn):
        """Fold a BatchNorm2d into the preceding conv kernel; returns
        (fused_kernel, fused_bias)."""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_kernel, fused_bias

    @torch.no_grad()
    def fuse(self):
        """Collapse the 3 branches into ONE depthwise 3x3 conv with bias.
        Mathematically exact for eval-mode BatchNorm (uses running stats)."""
        k3, b3 = self._fuse_conv_bn(self.conv3x3.weight, self.bn3x3)
        k1, b1 = self._fuse_conv_bn(self.conv1x1.weight, self.bn1x1)
        # identity branch: BN applied directly to x -- rewrite as a 3x3
        # depthwise kernel that is a scaled centre "spike" per channel.
        id_kernel = torch.zeros_like(k3)
        id_kernel[:, :, 1, 1] = 1.0
        k_id, b_id = self._fuse_conv_bn(id_kernel, self.bn_id)
        # pad the 1x1 kernel to 3x3 (zeros everywhere except centre)
        k1_padded = F.pad(k1, [1, 1, 1, 1])
        fused_kernel = k3 + k1_padded + k_id
        fused_bias = b3 + b1 + b_id

        fused_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1, groups=self.dim, bias=True)
        fused_conv.weight.copy_(fused_kernel)
        fused_conv.bias.copy_(fused_bias)
        self.fused = fused_conv

        # remove the now-unused multi-branch parameters so the parameter
        # count (and memory footprint) actually shrinks, as claimed by
        # structural re-parameterization -- not just an extra module
        # bolted on top.
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1, self.bn_id


# --------------------------------------------------------------------------
# 2. Squeeze-and-Excitation channel attention
# --------------------------------------------------------------------------
class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


# --------------------------------------------------------------------------
# 3. Channel mixer: pointwise MLP (ViT-FFN written with 1x1 convs)
# --------------------------------------------------------------------------
class ChannelMixer(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden = dim * mlp_ratio
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        y = F.gelu(self.bn1(self.pw1(x)))
        y = self.bn2(self.pw2(y))
        return x + y


# --------------------------------------------------------------------------
# 4. Full RepViT block = token mixer (+ optional SE) -> channel mixer
# --------------------------------------------------------------------------
class RepViTBlock(nn.Module):
    def __init__(self, dim, use_se=False, mlp_ratio=2):
        super().__init__()
        self.token_mixer = RepVGGDW(dim)
        self.se = SqueezeExcite(dim) if use_se else nn.Identity()
        self.channel_mixer = ChannelMixer(dim, mlp_ratio)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.se(x)
        x = self.channel_mixer(x)
        return x


def downsample(c_in, c_out):
    """Stride-2 conv used between stages (also BN, per RepViT's design of
    keeping every normalisation as a fusable BatchNorm)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.GELU(),
    )


# --------------------------------------------------------------------------
# 5. RepViT-mini network for 8x8x1 digit images, 10 classes
# --------------------------------------------------------------------------
class RepViTMini(nn.Module):
    def __init__(self, num_classes=10, widths=(24, 48), depths=(2, 2)):
        super().__init__()
        c1, c2 = widths
        d1, d2 = depths

        # stem: light-weight "early convolution" patchify, stride 1
        # (kept stride 1 because the input is only 8x8 -- unlike
        # ImageNet's 224x224, we cannot afford RepViT's usual 4x
        # downsampling stem or we would run out of pixels)
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )

        self.stage1 = nn.Sequential(
            *[RepViTBlock(c1, use_se=(i % 2 == 1)) for i in range(d1)]
        )
        self.down = downsample(c1, c2)
        self.stage2 = nn.Sequential(
            *[RepViTBlock(c2, use_se=(i % 2 == 1)) for i in range(d2)]
        )

        self.head = nn.Linear(c2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)

    @torch.no_grad()
    def fuse(self):
        """Fuse every RepVGGDW module in the network (train-time
        multi-branch -> inference-time single conv)."""
        for m in self.modules():
            if isinstance(m, RepVGGDW):
                m.fuse()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # ---- sanity check: fused module must be numerically equivalent ----
    torch.manual_seed(0)
    dw = RepVGGDW(16).eval()
    x = torch.randn(4, 16, 8, 8)
    with torch.no_grad():
        y_before = dw(x)
        dw.fuse()
        y_after = dw(x)
    max_err = (y_before - y_after).abs().max().item()
    print(f"[reparameterization check] max |before-after| = {max_err:.3e}")
    assert max_err < 1e-4, "Fusion is not numerically equivalent!"
    print("Structural re-parameterization verified: 3-branch training-time "
          "module and the fused single-conv inference-time module produce "
          "identical outputs.")
