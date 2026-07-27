# PyTorch compatibility port derived from CellFM retention components.
# This file is excluded from Morpho-FM's Apache-2.0 license; see
# THIRD_PARTY_NOTICES.md for the audited upstream revision and status.

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SRMSNorm(nn.Module):
    def __init__(self, emb_dims: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(emb_dims)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x_norm = torch.norm(x * self.scale, dim=-1, keepdim=True)
        x = x / torch.clamp(x_norm, min=self.eps)
        return x.to(dtype)


class MHRetention(nn.Module):
    def __init__(self, emb_dims: int, num_heads: int, lth: int | None = None) -> None:
        super().__init__()
        if emb_dims % num_heads != 0:
            raise ValueError("emb_dims must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dims = emb_dims // num_heads
        self.scale = math.sqrt(self.head_dims)
        beta = 1.0 if lth is None else (lth * 8) ** -0.25

        self.q_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.k_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)

        nn.init.xavier_normal_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

        self.inner_norm = SRMSNorm(self.head_dims)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        v_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            y = x
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        u = self.u_proj(x)

        b, l1, d = q.shape
        _, l2, _ = k.shape
        h = self.num_heads

        q = q.view(b, l1, h, self.head_dims).transpose(1, 2)
        k = k.view(b, l2, h, self.head_dims).transpose(1, 2)
        v = v.view(b, l2, h, self.head_dims).transpose(1, 2)
        u = u.view(b, l1, h, self.head_dims).transpose(1, 2)

        q = F.relu(q)
        k = F.relu(k)
        u = F.silu(u)

        if seq_mask is not None:
            q = q * seq_mask
        if attn_mask is not None:
            k = k * attn_mask
        if v_pos is not None:
            v = v * v_pos

        q = q / self.scale
        k = k / self.scale

        kv = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, kv)
        out = self.inner_norm(out)
        out = out * u
        out = out.transpose(1, 2).contiguous().view(b, l1, d)
        out = self.o_proj(out)
        return out


class GatedLinearUnit(nn.Module):
    def __init__(self, emb_dims: int, lth: int | None = None) -> None:
        super().__init__()
        beta = 1.0 if lth is None else (lth * 8) ** -0.25
        self.u_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.v_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        self.o_proj = nn.Linear(emb_dims, emb_dims, bias=False)
        nn.init.xavier_normal_(self.u_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.v_proj.weight, gain=beta)
        nn.init.xavier_normal_(self.o_proj.weight, gain=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(x)
        v = self.v_proj(x)
        out = self.o_proj(u * v)
        return out


class RetentionLayer(nn.Module):
    def __init__(
        self,
        emb_dims: int,
        num_heads: int,
        lth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MHRetention(emb_dims, num_heads, lth)
        self.ffn = GatedLinearUnit(emb_dims, lth)
        self.dropout = nn.Dropout(dropout)
        self.post_norm1 = nn.LayerNorm(emb_dims)
        self.post_norm2 = nn.LayerNorm(emb_dims)
        self.alpha = (2 * lth) ** 0.25

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        v_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.attn(x, y=y, v_pos=v_pos, attn_mask=attn_mask, seq_mask=seq_mask)
        x = self.post_norm1(x * self.alpha + self.dropout(out))
        out = self.ffn(x)
        x = self.post_norm2(x * self.alpha + self.dropout(out))
        return x
