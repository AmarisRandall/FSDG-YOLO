import copy
from typing import Tuple

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class AsymmetricDWBlock(nn.Module):
    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        pad = k // 2
        self.dw_h = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, k),
                stride=1,
                padding=(0, pad),
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
                padding=(pad, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.mix = Conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_h(x)
        x = self.dw_v(x)
        return self.mix(x)


class SpatialGate(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, stride=1, padding=pad, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        gate = self.act(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * gate


class FAGM(Detect):
    def __init__(
        self,
        nc: int = 80,
        p2_level: int = 0,
        reg_max: int = 16,
        end2end: bool = False,
        ch: Tuple[int, ...] = (),
    ):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        assert len(ch) > 0, "FAGM requires feature channels."
        assert 0 <= p2_level < len(ch), f"p2_level must be in [0, {len(ch) - 1}]"
        self.p2_level = p2_level
        c_p2 = ch[p2_level]
        c_mid = max(c_p2 // 2, 32)
        c_gate = max(c_p2 // 4, 16)
        self.p2_stem = nn.Sequential(
            Conv(c_p2, c_mid, 1),
            Conv(c_mid, c_mid, 3),
            AsymmetricDWBlock(c_mid, k=5),
            SpatialGate(k=7),
            Conv(c_mid, c_p2, 1),
        )
        self.align_gate = nn.Sequential(
            Conv(c_p2, c_gate, 1),
            nn.Conv2d(c_gate, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        if end2end:
            self.one2one_p2_stem = copy.deepcopy(self.p2_stem)
            self.one2one_align_gate = copy.deepcopy(self.align_gate)

    def _enhance_level(self, x: torch.Tensor, stem: nn.Module, gate_head: nn.Module) -> torch.Tensor:
        flame_feat = stem(x)
        gate = gate_head(flame_feat)
        return x + flame_feat * gate

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None:
            box_head = self.cv2
        if cls_head is None:
            cls_head = self.cv3
        x = list(x)
        x[self.p2_level] = self._enhance_level(x[self.p2_level], self.p2_stem, self.align_gate)
        return super().forward_head(x, box_head, cls_head)

    @property
    def one2many(self):
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            x_detach[self.p2_level] = self._enhance_level(
                x_detach[self.p2_level], self.one2one_p2_stem, self.one2one_align_gate
            )
            one2one = super().forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        return y if self.export else (y, preds)
