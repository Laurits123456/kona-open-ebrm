from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from kona.pretrain_data import DOMAIN_IDS
from kona.pretrain_model import KonaCharTokenizer, KonaEnergyModel, KonaPretrainConfig


@dataclass(frozen=True)
class KonaPairScore:
    domain_name: str
    energy: float
    plausibility_prob: float


class KonaPretrainScorer:
    def __init__(self, model: KonaEnergyModel, tokenizer: KonaCharTokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device: str = "cpu") -> "KonaPretrainScorer":
        payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        cfg = KonaPretrainConfig(**dict(payload["config"]))
        tokenizer_cfg = dict(payload.get("tokenizer") or {})
        tokenizer = KonaCharTokenizer(max_length=int(tokenizer_cfg.get("max_length", cfg.max_seq_len)))
        model = KonaEnergyModel(cfg, vocab_size=tokenizer.vocab_size).to(device)
        model.load_state_dict(dict(payload["model_state"]))
        model.eval()
        return cls(model, tokenizer, device)

    @torch.no_grad()
    def score_pair(self, domain_name: str, context_text: str, state_text: str) -> KonaPairScore:
        if domain_name not in DOMAIN_IDS:
            raise ValueError(f"Unknown domain: {domain_name}")
        ids, token_types, attention_mask = self.tokenizer.encode_pair(context_text, state_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        token_types_tensor = torch.tensor([token_types], dtype=torch.long, device=self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        domain_ids = torch.tensor([DOMAIN_IDS[domain_name]], dtype=torch.long, device=self.device)
        energy = float(self.model(input_ids, token_types_tensor, attention_mask_tensor, domain_ids).item())
        plausibility = float(torch.sigmoid(torch.tensor(-energy)).item())
        return KonaPairScore(domain_name=domain_name, energy=energy, plausibility_prob=plausibility)
