import torch
import torch.nn as nn
import torch.nn.functional as F

from .loftr_module.transformer import LocalFeatureTransformer


class Conv2d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


class CIM(nn.Module):
    """Feature Aggregation, Correlation Injection Module"""

    def __init__(self, config):
        super(CIM, self).__init__()

        self.block_dims = config["backbone"]["block_dims"]
        self.drop = config["fine"]["droprate"]

        self.fc32 = Conv2d_BN_Act(
            self.block_dims[-1], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc16 = Conv2d_BN_Act(
            self.block_dims[-2], self.block_dims[-1], 1, drop=self.drop
        )
        self.fc8 = Conv2d_BN_Act(
            self.block_dims[-3], self.block_dims[-1], 1, drop=self.drop
        )
        self.att32 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.att16 = Conv2d_BN_Act(
            self.block_dims[-1],
            self.block_dims[-1],
            1,
            act=nn.Sigmoid(),
            drop=self.drop,
        )
        self.dwconv16 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )
        self.dwconv8 = nn.Sequential(
            Conv2d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                ks=3,
                pad=1,
                groups=self.block_dims[-1],
                act=nn.GELU(),
            ),
            Conv2d_BN_Act(self.block_dims[-1], self.block_dims[-1], 1),
        )

        self.loftr_32 = LocalFeatureTransformer(config["neck"])

        # Covisibility branch
        self.covi_enabled = config["neck"]["covi_enabled"]
        if self.covi_enabled:
            self.covi_head = nn.Conv2d(self.block_dims[-1], 1, kernel_size=1, bias=True)
            self.covi_alpha_raw = nn.Parameter(torch.zeros(1))

    def _fuse_gate(self, att, covi_prob):
        """Residual-style gate fusion with bounded alpha via tanh."""
        alpha = torch.tanh(self.covi_alpha_raw)  # alpha ∈ (-1, 1)
        covi_mod = (2 * covi_prob - 1).expand_as(att)  # [-1, 1]
        # 乘法因子 ∈ (0, 2), gate_fused 始终为正
        return att * (1 + alpha * covi_mod)

    def forward(self, ms_feats, mask_c0=None, mask_c1=None):
        if len(ms_feats) == 3:  # same image shape
            f8, f16, f32 = ms_feats
            f32 = self.fc32(f32)

            f32_0, f32_1 = f32.chunk(2, dim=0)
            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)
            f32 = torch.cat([f32_0, f32_1], dim=0)

            # Covisibility branch
            if self.covi_enabled:
                covi_logits = self.covi_head(f32)              # [2B, 1, H/32, W/32]
                covi_logits_0, covi_logits_1 = covi_logits.chunk(2, dim=0)
                covi_prob = torch.sigmoid(covi_logits)          # [2B, 1, H/32, W/32]
            else:
                covi_logits_0 = covi_logits_1 = None

            att32 = self.att32(f32)                              # [2B, 256, H/32, W/32]

            if self.covi_enabled:
                gate_fused = self._fuse_gate(att32, covi_prob)
            else:
                gate_fused = att32

            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            gate_fused_up = F.interpolate(gate_fused, scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * gate_fused_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)

            feat_c0, feat_c1 = f8.chunk(2)

        elif len(ms_feats) == 6:  # diffirent image shape
            f8_0, f16_0, f32_0, f8_1, f16_1, f32_1 = ms_feats
            f32_0 = self.fc32(f32_0)
            f32_1 = self.fc32(f32_1)

            f32_0, f32_1 = self.loftr_32(f32_0, f32_1, mask_c0, mask_c1)

            # Image 0
            f8, f16, f32 = f8_0, f16_0, f32_0
            if self.covi_enabled:
                covi_logits_0 = self.covi_head(f32)              # [B, 1, H0/32, W0/32]
                covi_prob_0 = torch.sigmoid(covi_logits_0)
            else:
                covi_logits_0 = None

            att32_0 = self.att32(f32)
            if self.covi_enabled:
                gate_fused_0 = self._fuse_gate(att32_0, covi_prob_0)
            else:
                gate_fused_0 = att32_0

            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            gate_fused_up = F.interpolate(gate_fused_0, scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * gate_fused_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c0 = f8

            # Image 1
            f8, f16, f32 = f8_1, f16_1, f32_1
            if self.covi_enabled:
                covi_logits_1 = self.covi_head(f32)              # [B, 1, H1/32, W1/32]
                covi_prob_1 = torch.sigmoid(covi_logits_1)
            else:
                covi_logits_1 = None

            att32_1 = self.att32(f32)
            if self.covi_enabled:
                gate_fused_1 = self._fuse_gate(att32_1, covi_prob_1)
            else:
                gate_fused_1 = att32_1

            f32_up = F.interpolate(f32, scale_factor=2.0, mode="bilinear")
            gate_fused_up = F.interpolate(gate_fused_1, scale_factor=2.0, mode="bilinear")
            f16 = self.fc16(f16)
            f16 = self.dwconv16(f16 * gate_fused_up + f32_up)
            f16_up = F.interpolate(f16, scale_factor=2.0, mode="bilinear")
            att16_up = F.interpolate(self.att16(
                f16), scale_factor=2.0, mode="bilinear")
            f8 = self.fc8(f8)
            f8 = self.dwconv8(f8 * att16_up + f16_up)
            feat_c1 = f8

        if self.covi_enabled:
            return feat_c0, feat_c1, {"covi_logits_0": covi_logits_0, "covi_logits_1": covi_logits_1}
        else:
            return feat_c0, feat_c1, None
