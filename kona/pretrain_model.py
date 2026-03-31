from __future__ import annotations

import string
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KonaPretrainConfig:
    num_domains: int
    max_seq_len: int = 384
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 4
    ff_dim: int = 384
    dropout: float = 0.1


class KonaCharTokenizer:
    PAD = 0
    UNK = 1
    BOS = 2
    SEP = 3
    EOS = 4

    def __init__(self, max_length: int = 384) -> None:
        self.max_length = max_length
        charset = "\n\t" + string.printable
        self.char_to_id = {ch: idx + 5 for idx, ch in enumerate(dict.fromkeys(charset))}
        self.vocab_size = max(self.char_to_id.values(), default=4) + 1

    def _encode_text(self, text: str) -> list[int]:
        return [self.char_to_id.get(ch, self.UNK) for ch in text]

    def encode_pair(self, context_text: str, state_text: str) -> tuple[list[int], list[int], list[int]]:
        context_ids = self._encode_text(context_text)
        state_ids = self._encode_text(state_text)
        ids = [self.BOS] + context_ids + [self.SEP] + state_ids + [self.EOS]
        token_types = [0] * (len(context_ids) + 2) + [1] * (len(state_ids) + 1)
        if len(ids) > self.max_length:
            ids = ids[: self.max_length]
            token_types = token_types[: self.max_length]
            ids[-1] = self.EOS
        attention_mask = [1] * len(ids)
        pad = self.max_length - len(ids)
        if pad > 0:
            ids.extend([self.PAD] * pad)
            token_types.extend([0] * pad)
            attention_mask.extend([0] * pad)
        return ids, token_types, attention_mask


class KonaEnergyModel(nn.Module):
    def __init__(self, cfg: KonaPretrainConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=KonaCharTokenizer.PAD)
        self.type_embed = nn.Embedding(2, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.domain_embed = nn.Embedding(cfg.num_domains, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        attention_mask: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = (
            self.token_embed(input_ids)
            + self.type_embed(token_types)
            + self.pos_embed(positions)
            + self.domain_embed(domain_ids).unsqueeze(1)
        )
        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        cls_repr = x[:, 0]
        weights = attention_mask.unsqueeze(-1).to(x.dtype)
        mean_repr = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.norm(0.75 * cls_repr + 0.25 * mean_repr)
        return self.head(pooled).squeeze(-1)


def batch_margin_loss(energy: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos = energy[labels == 1]
    neg = energy[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return energy.new_tensor(0.0)
    diff = pos.unsqueeze(1) - neg.unsqueeze(0) + margin
    return F.softplus(diff).mean()
