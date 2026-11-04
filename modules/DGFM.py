from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autopad(kernel_size: int, dilation: int = 1) -> int:
    return dilation * (kernel_size - 1) // 2


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = _autopad(k, d)
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, act: bool = True) -> None:
        super().__init__()
        self.dw = ConvBNAct(channels, channels, k=kernel_size, s=1, g=channels, d=dilation, act=act)
        self.pw = ConvBNAct(channels, channels, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class AsymmetricDWBlock(nn.Module):
    def __init__(self, channels: int, k: int = 5, dilation: int = 1) -> None:
        super().__init__()
        pad_h = (0, dilation * (k - 1) // 2)
        pad_v = (dilation * (k - 1) // 2, 0)
        self.dw_h = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, k),
                stride=1,
                padding=pad_h,
                dilation=(1, dilation),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.dw_v = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(k, 1),
                stride=1,
                padding=pad_v,
                dilation=(dilation, 1),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.mix = ConvBNAct(channels, channels, k=1, s=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_h(x)
        x = self.dw_v(x)
        return self.mix(x)


class ECABlock(nn.Module):
    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv1d(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate


class SobelGradientGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.proj = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = torch.mean(x, dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return self.proj(grad)


class DynamicRouting(nn.Module):
    def __init__(self, channels: int, n_branches: int = 3, hidden_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(channels // hidden_ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, n_branches, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc2(self.act(self.fc1(self.pool(x))))
        return torch.softmax(logits, dim=1)


class HighFrequencyBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.grad_gate = SobelGradientGate(channels)
        self.b1 = DepthwiseSeparableConv(channels, kernel_size=3, dilation=1)
        self.b2 = DepthwiseSeparableConv(channels, kernel_size=3, dilation=2)
        self.b3 = AsymmetricDWBlock(channels, k=5, dilation=2)
        self.b4 = AsymmetricDWBlock(channels, k=7, dilation=2)
        self.branch_weights = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.spatial_gate = SpatialGate()
        self.out = ConvBNAct(channels, channels, k=1, s=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.grad_gate(x)
        xg = x * g
        branches = [self.b1(xg), self.b2(xg), self.b3(xg), self.b4(xg)]
        w = torch.softmax(self.branch_weights, dim=0)
        y = w[0] * branches[0] + w[1] * branches[1] + w[2] * branches[2] + w[3] * branches[3]
        y = self.spatial_gate(y)
        return self.out(y)


class LowFrequencyBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cur_proj = ConvBNAct(channels, channels, k=1, s=1, act=True)
        self.ctx_proj = ConvBNAct(channels, channels, k=1, s=1, act=True)
        self.fuse = ConvBNAct(channels * 2, channels, k=1, s=1, act=True)
        self.eca = ECABlock(channels, k_size=3)
        self.out = ConvBNAct(channels, channels, k=3, s=1, act=True)
    @staticmethod
    def low_pass(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        cur = self.cur_proj(self.low_pass(x))
        ctx = self.low_pass(x_ctx)
        ctx = F.interpolate(ctx, size=x.shape[-2:], mode="nearest")
        ctx = self.ctx_proj(ctx)
        y = self.fuse(torch.cat([cur, ctx], dim=1))
        y = self.eca(y)
        return self.out(y)


class PreserveGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_avg = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return self.proj(torch.cat([x_max, x_avg], dim=1))


class MixBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, k=1, s=1, act=True),
            DepthwiseSeparableConv(channels, kernel_size=5, dilation=1),
            ECABlock(channels, k_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DGFM(nn.Module):
    def __init__(self, channels: int, ctx_channels: int, reduction: int = 4, use_shortcut: bool = True) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pre = ConvBNAct(channels, hidden, k=1, s=1, act=True)
        self.ctx_pre = ConvBNAct(ctx_channels, hidden, k=1, s=1, act=True)
        self.high = HighFrequencyBranch(hidden)
        self.low = LowFrequencyBranch(hidden)
        self.mix = MixBranch(hidden)
        self.router = DynamicRouting(hidden, n_branches=3, hidden_ratio=4)
        self.preserve = PreserveGate(hidden)
        self.post = ConvBNAct(hidden, channels, k=1, s=1, act=False)
        self.use_shortcut = use_shortcut
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DGFM expects inputs as [x, x_ctx].")
        x, x_ctx = x
        identity = x
        z = self.pre(x)
        ctx = self.ctx_pre(x_ctx)
        h = self.high(z)
        l = self.low(z, ctx)
        m = self.mix(z)
        w = self.router(z)
        p = self.preserve(z)
        fused = w[:, 0:1] * h + w[:, 1:2] * l + w[:, 2:3] * m
        out = self.post(p * fused)
        if self.use_shortcut:
            out = identity + out
        return self.act(out)
