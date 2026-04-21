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

    def forward_mixed_res(self, query, key, value, q_mask=None, kv_mask=None):
        """
        Mixed-resolution attention for DCAT: supports Lq != Lkv.

        Args:
            query: [B, Hq, Wq, C] - query at original resolution
            key:   [B, Hk, Wk, C] - key at aggregated low resolution
            value: [B, Hk, Wk, C] - value at aggregated low resolution
            q_mask:  [B, Hq, Wq] or None - valid mask for query
            kv_mask: [B, Hk, Wk] or None - valid mask for key/value

        Returns:
            out: [B, Hq, Wq, nhead*dim] - output at query resolution
        """
        B = query.size(0)
        Hq, Wq = query.size(1), query.size(2)
        Hk, Wk = key.size(1), key.size(2)
        Lq = Hq * Wq
        Lk = Hk * Wk

        # Flatten spatial dims: [B, H, W, C] -> [B, L, nhead, dim]
        q = rearrange(query, "n h w (nhead d) -> n (h w) nhead d",
                      nhead=self.nhead, d=self.dim)  # [B, Lq, nhead, dim]
        k = rearrange(key,   "n h w (nhead d) -> n (h w) nhead d",
                      nhead=self.nhead, d=self.dim)  # [B, Lk, nhead, dim]
        v = rearrange(value, "n h w (nhead d) -> n (h w) nhead d",
                      nhead=self.nhead, d=self.dim)  # [B, Lk, nhead, dim]

        # Scaled Cosine Attention
        q = F.normalize(q, p=2, dim=3)
        k = F.normalize(k, p=2, dim=3)
        # QK: [B, nhead, Lq, Lk] via einsum
        QK = torch.einsum("nlhd,nshd->nhls", q, k)  # [B, nhead, Lq, Lk]
        s = 20.0

        # Apply mask before softmax
        if q_mask is not None or kv_mask is not None:
            # ASSERT: mask spatial dims must match corresponding tensor
            if q_mask is not None:
                assert q_mask.shape == (B, Hq, Wq), (
                    f"[mixed_res] q_mask shape {q_mask.shape} != query spatial ({B}, {Hq}, {Wq})"
                )
            if kv_mask is not None:
                assert kv_mask.shape == (B, Hk, Wk), (
                    f"[mixed_res] kv_mask shape {kv_mask.shape} != key spatial ({B}, {Hk}, {Wk})"
                )
            # Build additive mask: -inf for invalid positions
            mask = torch.zeros(B, 1, Lq, Lk, device=query.device, dtype=query.dtype)
            if q_mask is not None:
                # q_mask: [B, Hq, Wq] -> [B, 1, Lq, 1]
                qm = q_mask.reshape(B, 1, Lq, 1)
                mask = mask.masked_fill(~qm, float("-inf"))
            if kv_mask is not None:
                # kv_mask: [B, Hk, Wk] -> [B, 1, 1, Lk]
                kvm = kv_mask.reshape(B, 1, 1, Lk)
                mask = mask.masked_fill(~kvm, float("-inf"))
            A = torch.softmax(s * QK + mask, dim=3)
        else:
            A = torch.softmax(s * QK, dim=3)  # [B, nhead, Lq, Lk]

        # Weighted sum: [B, nhead, Lq, Lk] x [B, Lk, nhead, dim] -> [B, Lq, nhead, dim]
        out = torch.einsum("nhls,nshd->nlhd", A, v)  # [B, Lq, nhead, dim]

        return out


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
    source_mask=None,
):
    """
    Dynamic aggregation with matchability-guided weighting.
    This implements the core of CoMatch's Dynamic Covisibility-Aware Aggregation.

    IMPORTANT: This function now returns TRUE low-res kv (no interpolation back to query resolution).

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
        source_mask: [B, Hs, Ws] or None - valid mask for source features

    Returns:
        pooled_key_4d: [B, Hk, Wk, C] - aggregated and projected keys (TRUE low-res, NOT upsampled)
        pooled_value_4d: [B, Hk, Wk, C] - aggregated and projected values
        pooled_source_score: [B, 1, Hk, Wk] - max-pooled source matchability scores (low-res)
        kv_mask: [B, Hk, Wk] or None - valid mask for kv (pooled from source_mask via max pooling)
    """
    B, C, Hx, Wx = x.shape
    Hs, Ws = source_hw
    Hx, Wx = int(Hx), int(Wx)
    Hs, Ws = int(Hs), int(Ws)

    # ASSERT: source_matchability_score shape check
    if source_matchability_score is not None:
        assert source_matchability_score.shape == (B, 1, Hs, Ws), (
            f"[DCAT AGG] source_matchability_score shape {source_matchability_score.shape} "
            f"!= (B={B}, 1, Hs={Hs}, Ws={Ws})"
        )

    # Determine effective aggregation size (handle small feature maps)
    effective_agg_h = min(agg_size, Hs) if Hs > 0 else 1
    effective_agg_w = min(agg_size, Ws) if Ws > 0 else 1

    # Fallback to no aggregation if feature map is too small
    if Hs < effective_agg_h or Ws < effective_agg_w:
        # Return simple projection without aggregation (identity: Hk=Hs, Wk=Ws)
        source_proj = source.permute(0, 2, 3, 1)  # [B, Hs, Ws, C]
        source_proj = k_proj(source_proj)  # [B, Hs, Ws, C]
        pooled_key_4d = source_proj
        pooled_value_4d = source_proj  # same projection for value
        pooled_source_score = source_matchability_score  # [B, 1, Hs, Ws] or None
        kv_mask = None
        return pooled_key_4d, pooled_value_4d, pooled_source_score, kv_mask

    # Handle non-divisible dimensions with padding
    pad_h = (effective_agg_h - Hs % effective_agg_h) % effective_agg_h
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

    Hk = Hp // effective_agg_h
    Wk = Wp // effective_agg_w

    source_unfolded = source_unfolded.reshape(
        B, C, effective_agg_h, effective_agg_w, Hk, Wk
    )  # [B, C, agg_h, agg_w, Hk, Wk]
    source_unfolded = source_unfolded.permute(0, 4, 5, 1, 2, 3).reshape(
        B * Hk * Wk, C, effective_agg_h, effective_agg_w
    )  # [B*Hk*Wk, C, agg_h, agg_w]

    # Apply linear projection to source features before aggregation
    source_proj = k_proj(source_unfolded)  # [B*Hk*Wk, C, agg_h, agg_w]

    # Unfold source matchability scores (RAW scores for max pooling, before softmax)
    if source_score_padded is not None:
        source_score_unfolded_raw = F.unfold(
            source_score_padded,
            kernel_size=(effective_agg_h, effective_agg_w),
            stride=(effective_agg_h, effective_agg_w),
            padding=(0, 0),
        )  # [B, 1 * agg_h * agg_w, Hk * Wk]
        source_score_unfolded_raw = source_score_unfolded_raw.reshape(
            B, 1, effective_agg_h, effective_agg_w, Hk, Wk
        )  # [B, 1, agg_h, agg_w, Hk, Wk]
        source_score_unfolded_raw = source_score_unfolded_raw.permute(0, 4, 5, 1, 2, 3).reshape(
            B * Hk * Wk, 1, effective_agg_h, effective_agg_w
        )  # [B*Hk*Wk, 1, agg_h, agg_w]

        # For softmax weights: use raw scores
        source_score_for_softmax = source_score_unfolded_raw.reshape(
            B * Hk * Wk, effective_agg_h * effective_agg_w
        )  # [B*Hk*Wk, agg_h*agg_w]
        agg_weights = F.softmax(source_score_for_softmax, dim=-1)  # [B*Hk*Wk, agg_h*agg_w]
        agg_weights = agg_weights.reshape(
            B * Hk * Wk, 1, effective_agg_h, effective_agg_w
        )  # [B*Hk*Wk, 1, agg_h, agg_w]
    else:
        source_score_unfolded_raw = None
        agg_weights = None

    # Compute softmax weights from source matchability scores
    if agg_weights is not None:
        pass  # agg_weights already computed above
    else:
        numel = effective_agg_h * effective_agg_w
        agg_weights = torch.ones_like(source_proj[:, :1, :, :]) / numel

    # Weighted aggregation of keys (projected)
    weighted_keys = source_proj * agg_weights  # [B*Hk*Wk, C, agg_h, agg_w]
    pooled_key = weighted_keys.sum(dim=[2, 3])  # [B*Hk*Wk, C]

    # Weighted aggregation of values
    if source_score_unfolded_raw is not None:
        # Apply value projection first
        value_proj = v_proj(source_unfolded)  # [B*Hk*Wk, C, agg_h, agg_w]

        # Reshape source scores for multiplication
        source_scores = agg_weights.reshape(
            B * Hk * Wk, effective_agg_h * effective_agg_w
        )  # [B*Hk*Wk, agg_h*agg_w]

        # Weight values by source matchability before pooling
        value_proj_flat = value_proj.reshape(
            B * Hk * Wk, C, effective_agg_h * effective_agg_w
        )  # [B*Hk*Wk, C, agg_h*agg_w]

        weighted_values = value_proj_flat * source_scores.unsqueeze(1)  # [B*Hk*Wk, C, agg_h*agg_w]
        pooled_value = weighted_values.sum(dim=-1)  # [B*Hk*Wk, C]

        # Pool source matchability scores using max pooling on RAW scores
        source_score_raw_reshaped = source_score_unfolded_raw.reshape(
            B, Hk, Wk, effective_agg_h * effective_agg_w
        )  # [B, Hk, Wk, agg_h*agg_w]
        pooled_source_score = source_score_raw_reshaped.max(dim=-1, keepdim=True)[0]  # [B, Hk, Wk, 1]
    else:
        pooled_value = pooled_key  # No matchability = uniform aggregation
        pooled_source_score = None

    # Reshape back to 4D: [B, Hk, Wk, C]
    pooled_key_4d = pooled_key.reshape(B, Hk, Wk, C)
    pooled_value_4d = pooled_value.reshape(B, Hk, Wk, C)

    # If we had padding, crop to original size
    H_orig, W_orig = int(source_hw[0]), int(source_hw[1])
    Hk_orig = (H_orig + effective_agg_h - 1) // effective_agg_h  # ceil(H_orig/agg)
    Wk_orig = (W_orig + effective_agg_w - 1) // effective_agg_w  # ceil(W_orig/agg)

    if pad_h > 0 or pad_w > 0:
        pooled_key_4d = pooled_key_4d[:, :Hk_orig, :Wk_orig, :]
        pooled_value_4d = pooled_value_4d[:, :Hk_orig, :Wk_orig, :]
        if pooled_source_score is not None:
            pooled_source_score = pooled_source_score[:, :Hk_orig, :Wk_orig, :]

    # Assertion: verify output shapes
    assert pooled_key_4d.shape[:3] == (B, Hk_orig, Wk_orig), (
        f"[DCAT AGG] pooled_key_4d shape {pooled_key_4d.shape[:3]} "
        f"!= (B={B}, Hk={Hk_orig}, Wk={Wk_orig})"
    )
    assert pooled_value_4d.shape[:3] == (B, Hk_orig, Wk_orig), (
        f"[DCAT AGG] pooled_value_4d shape {pooled_value_4d.shape[:3]} "
        f"!= (B={B}, Hk={Hk_orig}, Wk={Wk_orig})"
    )

    # Reshape pooled_source_score to [B, 1, Hk, Wk] if available
    if pooled_source_score is not None:
        pooled_source_score = pooled_source_score.permute(0, 3, 1, 2)  # [B, 1, Hk, Wk]

    # === Generate kv_mask: max pooling of source_mask with same window/stride ===
    if source_mask is not None:
        source_mask_padded = source_mask
        if pad_h > 0 or pad_w > 0:
            source_mask_padded = F.pad(source_mask.float(), (0, pad_w, 0, pad_h)).bool()

        # Unfold mask with same kernel_size and stride as feature aggregation
        source_mask_unfolded = F.unfold(
            source_mask_padded.unsqueeze(1).float(),  # [B, 1, Hp, Wp]
            kernel_size=(effective_agg_h, effective_agg_w),
            stride=(effective_agg_h, effective_agg_w),
            padding=(0, 0),
        )  # [B, agg_h*agg_w, Hk*Wk]

        source_mask_unfolded = source_mask_unfolded.reshape(
            B, effective_agg_h, effective_agg_w, Hk, Wk
        )  # [B, agg_h, agg_w, Hk, Wk]
        source_mask_unfolded = source_mask_unfolded.permute(0, 3, 4, 1, 2).reshape(
            B * Hk * Wk, effective_agg_h, effective_agg_w
        )  # [B*Hk*Wk, agg_h, agg_w]

        # Max pooling: if ANY pixel in the window is valid, the pooled token is valid
        kv_mask_flat = source_mask_unfolded.max(dim=2)[0].max(dim=1)[0]  # [B*Hk*Wk]
        kv_mask = kv_mask_flat.reshape(B, Hk, Wk)  # [B, Hk, Wk]

        # Crop to original size if padding was added
        if pad_h > 0 or pad_w > 0:
            kv_mask = kv_mask[:, :Hk_orig, :Wk_orig]

        # ASSERT: kv_mask shape must match pooled_key_4d / pooled_value_4d
        assert kv_mask.shape == (B, Hk_orig, Wk_orig), (
            f"[DCAT AGG] kv_mask shape {kv_mask.shape} != (B={B}, Hk={Hk_orig}, Wk={Wk_orig})"
        )
    else:
        kv_mask = None

    return pooled_key_4d, pooled_value_4d, pooled_source_score, kv_mask


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

        # Apply first normalization
        x_norm = self.norm1(x.permute(0, 2, 3, 1))  # [B, H0, W0, C]
        source_norm = self.norm1(source.permute(0, 2, 3, 1))  # [B, H1, W1, C]

        if self.use_dcat and source_matchability_score is not None:
            # === DCAT: Dynamic Covisibility-Aware Aggregation (native mixed-res) ===
            # Use ORIGINAL resolution masks (query is at H0xW0, kv is at aggregated HkxWk)
            # DO NOT downsample x_mask further - it must match query spatial dimensions
            q_mask_for_dcat = x_mask  # [B, H0, W0] or None
            x_hw = (H0, W0) if x_hw is None else x_hw
            source_hw = (H1, W1) if source_hw is None else source_hw

            # Get query projection
            query = self.q_proj(x_norm)  # [B, H0, W0, C]

            # RoPE on query only (not on aggregated kv)
            if self.rope:
                query = self.rope_pos_enc(query)

            # Dynamic aggregation: returns TRUE low-res kv (no upsampling back to query resolution)
            pooled_key_4d, pooled_value_4d, pooled_source_score, kv_mask = _dynamic_aggregate_with_matchability(
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
                source_mask=source_mask,  # Pass original source_mask for kv_mask generation
            )

            Hk = pooled_key_4d.size(1)
            Wk = pooled_key_4d.size(2)

            # ASSERT: mixed-res attention shapes
            assert query.shape == (B, H0, W0, C), (
                f"[DCAT] query shape {query.shape} != ({B}, {H0}, {W0}, {C})"
            )
            assert pooled_key_4d.shape == (B, Hk, Wk, C), (
                f"[DCAT] pooled_key shape {pooled_key_4d.shape} != ({B}, {Hk}, {Wk}, {C})"
            )
            assert pooled_value_4d.shape == (B, Hk, Wk, C), (
                f"[DCAT] pooled_value shape {pooled_value_4d.shape} != ({B}, {Hk}, {Wk}, {C})"
            )
            # ASSERT: q_mask shape must match query spatial dims
            assert q_mask_for_dcat is None or q_mask_for_dcat.shape == (B, H0, W0), (
                f"[DCAT] q_mask shape {q_mask_for_dcat.shape if q_mask_for_dcat is not None else None} "
                f"!= ({B}, {H0}, {W0})"
            )
            # ASSERT: kv_mask shape must match pooled_key_4d spatial dims
            assert kv_mask is None or kv_mask.shape == (B, Hk, Wk), (
                f"[DCAT] kv_mask shape {kv_mask.shape if kv_mask is not None else None} "
                f"!= ({B}, {Hk}, {Wk})"
            )

            # Multi-head attention with NATIVE mixed resolution (Lq != Lkv)
            # forward_mixed_res handles [B,Hq,Wq,C] x [B,Hk,Wk,C] -> [B,Hq,Wq,C]
            m = self.attention.forward_mixed_res(
                query, pooled_key_4d, pooled_value_4d, q_mask=q_mask_for_dcat, kv_mask=kv_mask
            )

        else:
            # === Fallback: Simple aggregation without DCAT (legacy path) ===
            # Use downsampled masks since legacy attention expects same-resolution q and kv at 1/32
            # The legacy self.attention() uses [B,H,W,C] flatten to [B,L,nhead,dim] internally
            # It does NOT use x_mask_down internally, so these are only for the legacy attention call
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

            # Use same-res attention: query and key/value at SAME resolution (legacy behavior)
            # key/value at full resolution same as query
            query = self.q_proj(x_norm)  # [B, H0, W0, C]
            key = self.k_proj(source_norm)  # [B, H1, W1, C]
            value = self.v_proj(source_norm)  # [B, H1, W1, C]

            # RoPE on query (legacy: same as before)
            if self.rope:
                query = self.rope_pos_enc(query)

            # Legacy attention with same-resolution query/kv (Lq == Lkv)
            # Uses standard forward (NOT mixed-res)
            m = self.attention(
                query, key, value, q_mask=x_mask_down, kv_mask=source_mask_down
            )
            # m is [B, L, H, D], reshape back to [B, H0, W0, C]
            m = m.reshape(B, H0, W0, self.nhead * self.dim)

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

                    # ASSERT: predictor output shapes
                    assert logit0.shape == (B, 1, H0, W0), (
                        f"[Predictor] logit0 shape {logit0.shape} != ({B}, 1, {H0}, {W0})"
                    )
                    assert logit1.shape == (B, 1, H1, W1), (
                        f"[Predictor] logit1 shape {logit1.shape} != ({B}, 1, {H1}, {W1})"
                    )

                    # Convert to scores for feedback (sigmoid)
                    prev_score0 = torch.sigmoid(logit0)
                    prev_score1 = torch.sigmoid(logit1)

                    stage_matchability_logits.append((logit0, logit1))
                # i=7: no prediction, but still uses prev_score0/1 for feedback
            else:
                raise KeyError(f"Unknown layer type: {name}")

        return feat0, feat1, stage_matchability_logits
