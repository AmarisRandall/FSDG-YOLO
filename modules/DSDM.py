from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def autopad(k: int, p: Optional[int] = None, d: int = 1) -> int:
    if d > 1:
        k = d * (k - 1) + 1
    return k // 2 if p is None else p


class Conv(nn.Module):
    default_act = nn.SiLU()

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
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, d: int = 1, act: bool = True) -> None:
        super().__init__()
        self.dw = Conv(c1, c1, k=k, s=s, g=c1, d=d, act=act)
        self.pw = Conv(c1, c2, k=1, s=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16) -> None:
        super().__init__()
        hidden = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class FireBranch(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv(c_in, c_out, k=1, s=1),
            DWConv(c_out, c_out, k=3, s=1),
            DWConv(c_out, c_out, k=3, s=1, d=2),  # slightly larger receptive field without early downsample
            Conv(c_out, c_out, k=3, s=2),         # downsample after detail extraction
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmokeBranch(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.proj = Conv(c_in, c_out, k=1, s=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.context = nn.Sequential(
            DWConv(c_out, c_out, k=5, s=1),
            DWConv(c_out, c_out, k=3, s=1, d=2),
            Conv(c_out, c_out, k=1, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.pool(x)
        return self.context(x)


class DSDM(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        fire_ratio: float = 0.625,
        use_se: bool = True,
        use_shortcut: bool = True,
    ) -> None:
        super().__init__()
        fire_c = max(int(round(c2 * fire_ratio)), 8)
        smoke_c = max(c2 - fire_c, 8)
        total = fire_c + smoke_c
        if total != c2:
            smoke_c = max(c2 - fire_c, 8)
            total = fire_c + smoke_c
            if total != c2:
                fire_c = c2 - smoke_c
        self.fire_branch = FireBranch(c1, fire_c)
        self.smoke_branch = SmokeBranch(c1, smoke_c)
        self.use_se = use_se
        self.use_shortcut = use_shortcut
        self.attn = SEBlock(c2) if use_se else nn.Identity()
        self.fuse = nn.Sequential(
            Conv(c2, c2, k=1, s=1),
            DWConv(c2, c2, k=3, s=1),
        )
        self.short = Conv(c1, c2, k=3, s=2) if use_shortcut else nn.Identity()
        self.out = Conv(c2, c2, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fire_feat = self.fire_branch(x)
        smoke_feat = self.smoke_branch(x)
        y = torch.cat([fire_feat, smoke_feat], dim=1)
        y = self.attn(y)
        y = self.fuse(y)
        if self.use_shortcut:
            y = y + self.short(x)
        return self.out(y)
