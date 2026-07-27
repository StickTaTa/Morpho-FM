# PyTorch compatibility port derived from the CellFM model architecture.
# This file is excluded from Morpho-FM's Apache-2.0 license; see
# THIRD_PARTY_NOTICES.md for the audited upstream revision and status.

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from st_pipeline.model.cellfm_layers import RetentionLayer


def _pad_vocab_size(n_genes: int, multiple: int = 8) -> int:
    base = n_genes + 1
    return base + (-base) % multiple


class FFN(nn.Module):
    def __init__(self, in_dims: int, emb_dims: int, b: int = 256) -> None:
        super().__init__()
        self.w1 = nn.Linear(in_dims, b, bias=False)
        self.w3 = nn.Linear(b, b, bias=False)
        self.table = nn.Linear(b, emb_dims, bias=False)
        self.a = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        v = x.view(-1, d)
        v = F.leaky_relu(self.w1(v))
        v = self.w3(v) + v * self.a
        v = torch.softmax(v, dim=-1)
        v = self.table(v)
        return v.view(b, l, -1)


class ValueEncoder(nn.Module):
    def __init__(self, emb_dims: int) -> None:
        super().__init__()
        self.value_enc = FFN(1, emb_dims)
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = x.shape[:2]
        if x.dim() == 3 and x.shape[-1] == 2:
            unmask = x[..., :1]
            expr = x[..., 1:]
            expr_emb = self.value_enc(expr) * unmask + self.mask_emb * (1 - unmask)
        else:
            expr = x.view(b, l, 1)
            unmask = torch.ones_like(expr)
            expr_emb = self.value_enc(expr)
        return expr_emb, unmask


class ValueDecoder(nn.Module):
    def __init__(self, emb_dims: int, zero: bool = False) -> None:
        super().__init__()
        self.zero = zero
        self.w1 = nn.Linear(emb_dims, emb_dims, bias=False)
        self.w2 = nn.Linear(emb_dims, 1, bias=False)
        if self.zero:
            self.zero_logit = nn.Sequential(
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, emb_dims),
                nn.LeakyReLU(),
                nn.Linear(emb_dims, 1),
                nn.Sigmoid(),
            )

    def forward(self, expr_emb: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.w2(F.leaky_relu(self.w1(expr_emb)))
        pred = x.squeeze(-1)
        if not self.zero:
            return pred
        zero_prob = self.zero_logit(expr_emb).squeeze(-1)
        return pred, zero_prob


class CellwiseDecoder(nn.Module):
    def __init__(self, in_dims: int, emb_dims: int | None = None, zero: bool = False) -> None:
        super().__init__()
        emb_dims = emb_dims or in_dims
        self.map = nn.Linear(in_dims, emb_dims, bias=False)
        self.zero = zero
        if self.zero:
            self.zero_logit = nn.Linear(emb_dims, emb_dims)

    def forward(self, cell_emb: torch.Tensor, gene_emb: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query = torch.sigmoid(self.map(gene_emb))
        key = cell_emb.unsqueeze(-1)
        pred = torch.bmm(query, key).squeeze(-1)
        if not self.zero:
            return pred
        zero_query = self.zero_logit(gene_emb)
        zero_prob = torch.sigmoid(torch.bmm(zero_query, key)).squeeze(-1)
        return pred, zero_prob


class CellFMBackbone(nn.Module):
    def __init__(
        self,
        n_genes: int,
        emb_dims: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        add_zero: bool = False,
        pad_zero: bool = False,
    ) -> None:
        super().__init__()
        self.num_genes = n_genes
        self.emb_dims = emb_dims
        self.num_layers = num_layers
        self.pad_zero = pad_zero
        self.add_zero = add_zero

        vocab_size = _pad_vocab_size(n_genes)
        self.vocab_size = vocab_size
        self.gene_emb = nn.Parameter(torch.empty(vocab_size, emb_dims))
        self.cls_token = nn.Parameter(torch.empty(1, 1, emb_dims))
        self.zero_emb = nn.Parameter(torch.zeros(1, 1, emb_dims))

        nn.init.xavier_normal_(self.gene_emb, gain=0.5)
        nn.init.xavier_normal_(self.cls_token, gain=0.5)
        with torch.no_grad():
            self.gene_emb[0].zero_()

        self.value_enc = ValueEncoder(emb_dims)
        self.encoder = nn.ModuleList(
            [
                RetentionLayer(
                    emb_dims,
                    num_heads,
                    num_layers,
                    dropout * i / max(num_layers, 1),
                )
                for i in range(num_layers)
            ]
        )
        self.value_dec = ValueDecoder(emb_dims, zero=add_zero)
        self.cellwise_dec = CellwiseDecoder(emb_dims, emb_dims, zero=add_zero)

    def forward_prompt(
        self,
        prompt_token: torch.Tensor,
        gene_ids: torch.Tensor,
        use_retention: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_token.dim() == 2:
            prompt_token = prompt_token.unsqueeze(1)
        gene_emb = F.embedding(gene_ids, self.gene_emb)
        cls = prompt_token + self.cls_token
        tokens = torch.cat([cls, gene_emb], dim=1)
        if use_retention:
            for layer in self.encoder:
                tokens = layer(tokens)
        cls_out = tokens[:, 0]
        gene_out = tokens[:, 1:]
        return cls_out, gene_out
