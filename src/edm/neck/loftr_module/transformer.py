from torch.nn import Module
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(128, 128), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/32 featmap, the max length of 128 corresponds to 4096 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(
            0).float().unsqueeze(-1)  # [H, 1]
        j_position = torch.ones(max_shape).cumsum(
            1).float().unsqueeze(-1)  # [W, 1]

        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = (
            npe[0],
            npe[1],
            npe[2],
            npe[3],
        )
        i_position, j_position = (
            i_position * train_res_H / test_res_H,
            j_position * train_res_W / test_res_W,
        )

        div_term = torch.exp(
            torch.arange(0, d_model // 4, 1).float()
            * (-math.log(10000.0) / (d_model // 4))
        )
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(
            *max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32
        )
        cos = torch.zeros(
            *max_shape, d_model // 2, dtype=torch.float16 if ropefp16 else torch.float32
        )

        sin[:, :, 0::2] = (
            torch.sin(i_position * div_term).half()
            if ropefp16
            else torch.sin(i_position * div_term)
        )
        sin[:, :, 1::2] = (
            torch.sin(j_position * div_term).half()
            if ropefp16
            else torch.sin(j_position * div_term)
        )
        cos[:, :, 0::2] = (
            torch.cos(i_position * div_term).half()
            if ropefp16
            else torch.cos(i_position * div_term)
        )
        cos[:, :, 1::2] = (
            torch.cos(j_position * div_term).half()
            if ropefp16
            else torch.cos(j_position * div_term)
        )

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer(
            "sin", sin.unsqueeze(0), persistent=False
        )  # [1, H, W, C//2]
        self.register_buffer(
            "cos", cos.unsqueeze(0), persistent=False
        )  # [1, H, W, C//2]

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        return (x * self.cos[:, : x.size(1), : x.size(2), :]) + (
            self.rotate_half(x) * self.sin[:, : x.size(1), : x.size(2), :]
        )

    def rotate_half(self, x):
        # x = x.unflatten(-1, (-1, 2))
        a, b, c, d = x.shape
        x = x.reshape(a, b, c, d // 2, 2)

        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""


def crop_feature(query, key, value, x_mask, source_mask):
    mask_h0, mask_w0, mask_h1, mask_w1 = (
        x_mask[0].sum(-2)[0],
        x_mask[0].sum(-1)[0],
        source_mask[0].sum(-2)[0],
        source_mask[0].sum(-1)[0],
    )
    query = query[:, :mask_h0, :mask_w0, :]
    key = key[:, :mask_h1, :mask_w1, :]
    value = value[:, :mask_h1, :mask_w1, :]
    return query, key, value, mask_h0, mask_w0


def pad_feature(m, mask_h0, mask_w0, x_mask):
    bs, hw, nhead, dim = m.shape
    m = m.view(bs, mask_h0, mask_w0, nhead, dim)
    if mask_h0 != x_mask.size(-2):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0),
                    x_mask.size(-2) - mask_h0,
                    x_mask.size(-1),
                    nhead,
                    dim,
                    device=m.device,
                    dtype=m.dtype,
                ),
            ],
            dim=1,
        )
    elif mask_w0 != x_mask.size(-1):
        m = torch.cat(
            [
                m,
                torch.zeros(
                    m.size(0),
                    x_mask.size(-2),
                    x_mask.size(-1) - mask_w0,
                    nhead,
                    dim,
                    device=m.device,
                    dtype=m.dtype,
                ),
            ],
            dim=2,
        )
    return m


class Attention(Module):
    def __init__(self, nhead=8, dim=256, re=False):
        super().__init__()

        self.nhead = nhead
        self.dim = dim

    def attention(self, query, key, value, q_mask=None, kv_mask=None):
        assert (
            q_mask is None and kv_mask is None
        ), "Not support generalized attention mask yet."
        # Scaled Cosine Attention
        # Refer to "Query-key normalization for transformers" and "https://kexue.fm/archives/9859"
        query = F.normalize(query, p=2, dim=3)
        key = F.normalize(key, p=2, dim=3)
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        s = 20.0
        A = torch.softmax(s * QK, dim=2)

        out = torch.einsum("nlsh,nshd->nlhd", A, value)
        return out

    def _forward(self, query, key, value, q_mask=None, kv_mask=None):
        if q_mask is not None:
            query, key, value, mask_h0, mask_w0 = crop_feature(
                query, key, value, q_mask, kv_mask
            )

        query, key, value = map(
            lambda x: rearrange(
                x,
                "n h w (nhead d) -> n (h w) nhead d",
                nhead=self.nhead,
                d=self.dim,
            ),
            [query, key, value],
        )

        m = self.attention(query, key, value, q_mask=None, kv_mask=None)

        if q_mask is not None:
            m = pad_feature(m, mask_h0, mask_w0, q_mask)

        return m

    def forward(self, query, key, value, q_mask=None, kv_mask=None):
        """
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = query.size(0)
        if bs == 1 or q_mask is None:
            m = self._forward(query, key, value,
                              q_mask=q_mask, kv_mask=kv_mask)
        else:  # for faster trainning with padding mask while batch size > 1
            m_list = []
            for i in range(bs):
                m_list.append(
                    self._forward(
                        query[i: i + 1],
                        key[i: i + 1],
                        value[i: i + 1],
                        q_mask=q_mask[i: i + 1],
                        kv_mask=kv_mask[i: i + 1],
                    )
                )
            m = torch.cat(m_list, dim=0)
        return m


# ============================================
# Helper Functions for DCAT
# ============================================

def flatten_to_map(x, hw):
    """Convert [B, HW, C] to [B, C, H, W]"""
    B, HW, C = x.shape
    H, W = hw
    return x.transpose(1, 2).reshape(B, C, H, W)


def map_to_flatten(x):
    """Convert [B, C, H, W] to [B, HW, C]"""
    B, C, H, W = x.shape
    return x.reshape(B, C, H * W).transpose(1, 2)


def _dynamic_aggregate_with_matchability(
    x,
    source,
    x_matchability_score,
    source_matchability_score,
    x_hw,
    source_hw,
    agg_size,
    dim,
    nhead,
    k_proj,
    v_proj,
):
    """
    Dynamic aggregation with matchability-guided weighting.
    This implements the core of CoMatch's Dynamic Covisibility-Aware Aggregation.

    Args:
        x: [B, C, Hx, Wx] - query feature map
        source: [B, C, Hs, Ws] - source feature map for key/value
        x_matchability_score: [B, 1, Hx, Wx] or None - query matchability
        source_matchability_score: [B, 1, Hs, Ws] or None - source matchability
        x_hw: (Hx, Wx)
        source_hw: (Hs, Ws)
        agg_size: aggregation window size
        dim: feature dimension per head
        nhead: number of attention heads
        k_proj: Linear projection for keys
        v_proj: Linear projection for values

    Returns:
        pooled_key: [B, Hx*Wx, C] - aggregated and projected keys
        pooled_value: [B, Hx*Wx, C] - aggregated and projected values (weighted by pooled matchability)
        pooled_source_score: [B, 1, Hx, Wx] - pooled source matchability scores (for later weighting)
    """
    B, C, Hx, Wx = x.shape
    Hs, Ws = source_hw
    Hx, Wx = int(Hx), int(Wx)
    Hs, Ws = int(Hs), int(Ws)

    # Determine effective aggregation size (handle small feature maps)
    effective_agg_h = min(agg_size, Hx) if Hx > 0 else 1
    effective_agg_w = min(agg_size, Wx) if Wx > 0 else 1

    # Fallback to no aggregation if feature map is too small
    if Hx < effective_agg_h or Ws < effective_agg_w:
        # Return simple projection without aggregation
        source_proj = v_proj(source.permute(0, 2, 3, 1)).reshape(B, C, Hx * Wx).transpose(1, 2)
        return (
            x.reshape(B, C, Hx * Wx).transpose(1, 2),
            source_proj,
            None,
        )

    # Handle non-divisible dimensions with padding
    pad_h = (effective_agg_h - Hx % effective_agg_h) % effective_agg_h
    pad_w = (effective_agg_w - Ws % effective_agg_w) % effective_agg_w

    source_padded = source
    source_score_padded = source_matchability_score

    if pad_h > 0 or pad_w > 0:
        source_padded = F.pad(source, (0, pad_w, 0, pad_h))
        if source_matchability_score is not None:
            source_score_padded = F.pad(source_matchability_score, (0, pad_w, 0, pad_h))

    Hp = source_padded.size(2)
    Wp = source_padded.size(3)

    # Unfold source into local windows for aggregation
    source_unfolded = F.unfold(
        source_padded,
        kernel_size=(effective_agg_h, effective_agg_w),
        stride=(effective_agg_h, effective_agg_w),
        padding=(0, 0),
    )  # [B, C * agg_h * agg_w, Hp//agg_h * Wp//agg_w]

    Hp_out = Hp // effective_agg_h
    Wp_out = Wp // effective_agg_w

    source_unfolded = source_unfolded.reshape(
        B, C, effective_agg_h, effective_agg_w, Hp_out, Wp_out
    )  # [B, C, agg_h, agg_w, Hp_out, Wp_out]
    source_unfolded = source_unfolded.permute(0, 4, 5, 1, 2, 3).reshape(
        B * Hp_out * Wp_out, C, effective_agg_h, effective_agg_w
    )  # [B*Hp_out*Wp_out, C, agg_h, agg_w]

    # Apply linear projection to source features before aggregation
    source_proj = k_proj(source_unfolded)  # [B*Hp_out*Wp_out, C, agg_h, agg_w]

    # Unfold source matchability scores (RAW scores for max pooling, before softmax)
    if source_score_padded is not None:
        source_score_unfolded_raw = F.unfold(
            source_score_padded,
            kernel_size=(effective_agg_h, effective_agg_w),
            stride=(effective_agg_h, effective_agg_w),
            padding=(0, 0),
        )  # [B, 1 * agg_h * agg_w, Hp_out * Wp_out]
        source_score_unfolded_raw = source_score_unfolded_raw.reshape(
            B, 1, effective_agg_h, effective_agg_w, Hp_out, Wp_out
        )  # [B, 1, agg_h, agg_w, Hp_out, Wp_out]
        source_score_unfolded_raw = source_score_unfolded_raw.permute(0, 4, 5, 1, 2, 3).reshape(
            B * Hp_out * Wp_out, 1, effective_agg_h, effective_agg_w
        )  # [B*Hp_out*Wp_out, 1, agg_h, agg_w]

        # For softmax weights: use raw scores
        source_score_for_softmax = source_score_unfolded_raw.reshape(
            B * Hp_out * Wp_out, effective_agg_h * effective_agg_w
        )  # [B*Hp_out*Wp_out, agg_h*agg_w]
        agg_weights = F.softmax(source_score_for_softmax, dim=-1)  # [B*Hp_out*Wp_out, agg_h*agg_w]
        agg_weights = agg_weights.reshape(
            B * Hp_out * Wp_out, 1, effective_agg_h, effective_agg_w
        )  # [B*Hp_out*Wp_out, 1, agg_h, agg_w]
    else:
        source_score_unfolded_raw = None
        agg_weights = None

    # Compute softmax weights from source matchability scores
    if agg_weights is not None:
        # Uniform weights if no matchability scores
        pass  # agg_weights already computed above
    else:
        numel = effective_agg_h * effective_agg_w
        agg_weights = torch.ones_like(source_proj[:, :1, :, :]) / numel

    # Weighted aggregation of keys (projected)
    weighted_keys = source_proj * agg_weights  # [B*Hp_out*Wp_out, C, agg_h, agg_w]
    pooled_key = weighted_keys.sum(dim=[2, 3])  # [B*Hp_out*Wp_out, C]

    # Weighted aggregation of values: value is weighted by matchability BEFORE pooling
    if source_score_unfolded_raw is not None:
        # Apply value projection first
        value_proj = v_proj(source_unfolded)  # [B*Hp_out*Wp_out, C, agg_h, agg_w]

        # Reshape source scores for multiplication (use softmax weights)
        source_scores = agg_weights.reshape(
            B * Hp_out * Wp_out, effective_agg_h * effective_agg_w
        )  # [B*Hp_out*Wp_out, agg_h*agg_w]

        # Weight values by source matchability before pooling
        value_proj_flat = value_proj.reshape(
            B * Hp_out * Wp_out, C, effective_agg_h * effective_agg_w
        )  # [B*Hp_out*Wp_out, C, agg_h*agg_w]

        # Multiply values by matchability scores (weighted value aggregation)
        weighted_values = value_proj_flat * source_scores.unsqueeze(1)  # [B*Hp_out*Wp_out, C, agg_h*agg_w]
        pooled_value = weighted_values.sum(dim=-1)  # [B*Hp_out*Wp_out, C]

        # Pool source matchability scores using max pooling on RAW scores (aligned with CoMatch)
        # CoMatch: pooled_source_matchability_score = self.max_pool(source_matchability_score)
        # We apply max pooling on the raw (pre-softmax) unfolded scores
        source_score_raw_reshaped = source_score_unfolded_raw.reshape(
            B, Hp_out, Wp_out, effective_agg_h * effective_agg_w
        )  # [B, Hp_out, Wp_out, agg_h*agg_w]
        pooled_source_score = source_score_raw_reshaped.max(dim=-1, keepdim=True)[0]  # [B, Hp_out, Wp_out, 1]
        pooled_source_score = pooled_source_score.reshape(B * Hp_out * Wp_out, 1)  # [B*Hp_out*Wp_out, 1]
    else:
        pooled_value = pooled_key  # No matchability = uniform aggregation
        pooled_source_score = None

    # Reshape back
    pooled_key = pooled_key.reshape(B, Hp_out * Wp_out, C)  # [B, Hp_out*Wp_out, C]
    pooled_value = pooled_value.reshape(B, Hp_out * Wp_out, C)  # [B, Hp_out*Wp_out, C]

    # If we had padding, crop to original size
    if pad_h > 0 or pad_w > 0:
        pooled_key = pooled_key[:, : Hx * Wx, :]
        pooled_value = pooled_value[:, : Hx * Wx, :]
        if pooled_source_score is not None:
            pooled_source_score = pooled_source_score[:, : Hx * Wx, :]

    # Reshape pooled_source_score to [B, 1, Hx, Wx] if available
    if pooled_source_score is not None:
        pooled_source_score = pooled_source_score.reshape(B, 1, Hx, Wx)

    return pooled_key, pooled_value, pooled_source_score


class AG_RoPE_EncoderLayer(nn.Module):
    """
    Attention-guided RoPE Encoder Layer with Matchability-Aware Aggregation.

    This layer implements CoMatch's Dynamic Covisibility-Aware Transformer (DCAT)
    mechanism at EDM's 1/32 coarse resolution.
    """

    def __init__(
        self,
        d_model,
        nhead,
        agg_size=2,
        rope=False,
        npe=None,
        use_dcat=True,
    ):
        super(AG_RoPE_EncoderLayer, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size = agg_size
        self.rope = rope
        self.use_dcat = use_dcat

        # Mask for downsampling (used when handling different resolutions)
        self.mask_max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # RoPE positional encoding
        self.rope_pos_enc = RoPEPositionEncodingSine(
            d_model, max_shape=(128, 128), npe=npe, ropefp16=True
        )

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(self.nhead, self.dim)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x,
        source,
        x_mask=None,
        source_mask=None,
        x_matchability_score=None,
        source_matchability_score=None,
        x_hw=None,
        source_hw=None,
    ):
        """
        Args:
            x (torch.Tensor): [B, C, H0, W0] - query feature map
            source (torch.Tensor): [B, C, H1, W1] - source feature map for key/value
            x_mask (torch.Tensor): [B, H0, W0] (optional) - valid mask for query
            source_mask (torch.Tensor): [B, H1, W1] (optional) - valid mask for source
            x_matchability_score (torch.Tensor): [B, 1, H0, W0] (optional) - query matchability
            source_matchability_score (torch.Tensor): [B, 1, H1, W1] (optional) - source matchability
            x_hw: (H0, W0) - tuple of query dimensions
            source_hw: (H1, W1) - tuple of source dimensions
        """
        B, C, H0, W0 = x.shape
        H1, W1 = source.shape[2], source.shape[3]

        # Handle mask downsampling for cross-resolution
        if x_mask is not None and source_mask is not None:
            x_mask_down = self.mask_max_pool(
                self.mask_max_pool(x_mask.float())
            ).bool()
            source_mask_down = self.mask_max_pool(
                self.mask_max_pool(source_mask.float())
            ).bool()
        else:
            x_mask_down = None
            source_mask_down = None

        # Apply first normalization
        x_norm = self.norm1(x.permute(0, 2, 3, 1))  # [B, H0, W0, C]
        source_norm = self.norm1(source.permute(0, 2, 3, 1))  # [B, H1, W1, C]

        if self.use_dcat and source_matchability_score is not None:
            # === DCAT: Dynamic Covisibility-Aware Aggregation ===
            x_hw = (H0, W0) if x_hw is None else x_hw
            source_hw = (H1, W1) if source_hw is None else source_hw

            # Get query projection
            query = self.q_proj(x_norm)  # [B, H0, W0, C]

            # RoPE on query
            if self.rope:
                query = self.rope_pos_enc(query)

            # Dynamic aggregation with matchability guidance
            # Projections are applied INSIDE the aggregation function
            pooled_key, pooled_value, pooled_source_score = _dynamic_aggregate_with_matchability(
                x,
                source,
                x_matchability_score,
                source_matchability_score,
                x_hw,
                source_hw,
                self.agg_size,
                self.dim,
                self.nhead,
                self.k_proj,
                self.v_proj,
            )

            # Reshape pooled features
            pooled_key = pooled_key.reshape(B, H0, W0, C)
            pooled_value = pooled_value.reshape(B, H0, W0, C)

            # Apply RoPE to pooled key
            if self.rope:
                pooled_key = self.rope_pos_enc(pooled_key)

            # Apply value weighting with pooled source matchability (after pooling)
            # CoMatch: pooled_value = pooled_value * pooled_source_score
            if pooled_source_score is not None:
                # pooled_source_score is [B, 1, H0, W0]
                value_weight = pooled_source_score.permute(0, 2, 3, 1)  # [B, H0, W0, 1]
                pooled_value = pooled_value * value_weight  # Pure multiplicative weighting

            # Multi-head attention with pooled key/value
            m = self.attention(
                query, pooled_key, pooled_value, q_mask=x_mask_down, kv_mask=None
            )
        else:
            # === Fallback: Simple aggregation without DCAT ===
            # Average pooling of source for key/value
            source_pooled = F.avg_pool2d(source, kernel_size=self.agg_size, stride=self.agg_size)
            Hs_pool, Ws_pool = source_pooled.shape[2], source_pooled.shape[3]

            # Query and key projections
            query = self.q_proj(x_norm)  # [B, H0, W0, C]
            key = self.k_proj(source_norm)  # [B, H1, W1, C]
            value = self.v_proj(source_norm)  # [B, H1, W1, C]

            # RoPE
            if self.rope:
                query = self.rope_pos_enc(query)
                key = self.rope_pos_enc(key)

            # Reshape for attention
            query = query.reshape(B, H0 * W0, C)
            key = key.reshape(B, H1 * W1, C)
            value = value.reshape(B, H1 * W1, C)

            # Attention
            m = self.attention(
                query, key, value, q_mask=x_mask_down, kv_mask=source_mask_down
            )
            m = m.reshape(B, H0, W0, C)

        # Merge multi-head output
        m = self.merge(m.reshape(B, -1, self.nhead * self.dim))  # [B, H0*W0, C]
        m = m.reshape(B, H0, W0, C).permute(0, 3, 1, 2)  # [B, C, H0, W0]

        # Residual connection
        out = x + m

        # Feed-forward network
        out = self.mlp(out.permute(0, 2, 3, 1))  # [B, H0, W0, C]
        out = self.norm2(out).permute(0, 3, 1, 2)  # [B, C, H0, W0]

        return out


class MatchabilityPredictor(nn.Module):
    """
    Matchability Predictor for DCAT.

    Each predictor takes the cross-attended feature and predicts a matchability score.
    There are 4 independent predictors, one for each self-cross stage.
    """

    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.predictor = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            # Pointwise projection to 1 channel
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] - feature map from current stage

        Returns:
            logit: [B, 1, H, W] - matchability logit (before sigmoid)
        """
        return self.predictor(x)


class LocalFeatureTransformer(nn.Module):
    """
    Local Feature Transformer with Dynamic Covisibility-Aware Transformer (DCAT).

    This implements CoMatch's DCAT mechanism adapted for EDM's 1/32 coarse resolution:
    - 8-layer transformer: ['self', 'cross'] * 4
    - 3 Matchability Predictors (aligned with CoMatch: i=1,3,5 predict; i=7 does not)
    - Internal feedback loop: predictor output feeds into next stage
    - Matchability-aware aggregation in each transformer layer
    """

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.layer_names = config["layer_names"]
        self.agg_size = config.get("agg_size0", 2)
        self.rope = config["rope"]

        # DCAT configuration
        self.dcat_enabled = config.get("dcat", {}).get("enabled", True)
        self.num_stages = config.get("dcat", {}).get("num_stages", 3)  # 3 predictor outputs

        # Self layer (with RoPE)
        self_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            self.agg_size,
            config["rope"],
            config["npe"],
            use_dcat=self.dcat_enabled,
        )

        # Cross layer (without RoPE)
        cross_layer = AG_RoPE_EncoderLayer(
            config["d_model"],
            config["nhead"],
            self.agg_size,
            False,  # No RoPE for cross attention
            config["npe"],
            use_dcat=self.dcat_enabled,
        )

        self.layers = nn.ModuleList(
            [
                (
                    copy.deepcopy(self_layer)
                    if _ == "self"
                    else copy.deepcopy(cross_layer)
                )
                for _ in self.layer_names
            ]
        )

        # 4 independent Matchability Predictors
        self.predictors = nn.ModuleList(
            [
                MatchabilityPredictor(
                    in_channels=config["d_model"],
                    hidden_channels=config.get("dcat", {}).get("predictor_dim", 256),
                )
                for _ in range(self.num_stages)
            ]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        DCAT-enabled forward pass (CoMatch-aligned: 8-layer transformer, 3 predictor outputs).

        Data flow:
        - layer i=0 (self): no score feedback (first self)
        - layer i=1 (cross): predict -> score0 -> feedback to i=2,3
        - layer i=2 (self): use score0 for feedback
        - layer i=3 (cross): predict -> score1 -> feedback to i=4,5
        - layer i=4 (self): use score1 for feedback
        - layer i=5 (cross): predict -> score2 -> feedback to i=6,7
        - layer i=6 (self): use score2 for feedback
        - layer i=7 (cross): use score2 for feedback, NO prediction

        Args:
            feat0 (torch.Tensor): [B, C, H0, W0]
            feat1 (torch.Tensor): [B, C, H1, W1]
            mask0 (torch.Tensor): [B, H0, W0] (optional)
            mask1 (torch.Tensor): [B, H1, W1] (optional)
            data (dict): additional data

        Returns:
            feat0, feat1: updated features
            stage_matchability_logits: list of 3 tuples (logit0, logit1) from cross i=1,3,5
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape

        hw0 = (H0, W0)
        hw1 = (H1, W1)

        # Initialize matchability scores (None for first stage)
        prev_score0 = None
        prev_score1 = None

        # Store logits from each cross stage (i=1,3,5 only)
        stage_matchability_logits = []

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if name == "self":
                # Self attention: each image attends to itself
                feat0 = layer(
                    feat0,
                    feat0,
                    x_mask=mask0,
                    source_mask=mask0,
                    x_matchability_score=prev_score0,
                    source_matchability_score=prev_score0,
                    x_hw=hw0,
                    source_hw=hw0,
                )
                feat1 = layer(
                    feat1,
                    feat1,
                    x_mask=mask1,
                    source_mask=mask1,
                    x_matchability_score=prev_score1,
                    source_matchability_score=prev_score1,
                    x_hw=hw1,
                    source_hw=hw1,
                )

            elif name == "cross":
                # Cross attention: must cache previous state to avoid sequential pollution
                feat0_prev, feat1_prev = feat0, feat1

                # Parallel cross attention (avoid using updated feat0 for feat1)
                feat0 = layer(
                    feat0_prev,
                    feat1_prev,
                    x_mask=mask0,
                    source_mask=mask1,
                    x_matchability_score=prev_score0,
                    source_matchability_score=prev_score1,
                    x_hw=hw0,
                    source_hw=hw1,
                )
                feat1 = layer(
                    feat1_prev,
                    feat0_prev,
                    x_mask=mask1,
                    source_mask=mask0,
                    x_matchability_score=prev_score1,
                    source_matchability_score=prev_score0,
                    x_hw=hw1,
                    source_hw=hw0,
                )

                # CoMatch-aligned: predict only at cross i=1,3,5 (not at i=7)
                # i=1 -> predictor[0], i=3 -> predictor[1], i=5 -> predictor[2]
                if i in {1, 3, 5}:
                    predictor_idx = i // 2  # 1//2=0, 3//2=1, 5//2=2
                    logit0 = self.predictors[predictor_idx](feat0)  # [B, 1, H0, W0]
                    logit1 = self.predictors[predictor_idx](feat1)  # [B, 1, H1, W1]

                    # Convert to scores for feedback (sigmoid)
                    prev_score0 = torch.sigmoid(logit0)
                    prev_score1 = torch.sigmoid(logit1)

                    stage_matchability_logits.append((logit0, logit1))
                # i=7: no prediction, but still uses prev_score0/1 for feedback
            else:
                raise KeyError(f"Unknown layer type: {name}")

        return feat0, feat1, stage_matchability_logits
