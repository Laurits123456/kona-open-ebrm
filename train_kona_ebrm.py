from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kona.pretrain_data import DOMAIN_ORDER, KonaPretrainDataset
from kona.pretrain_model import (
    KonaCharTokenizer,
    KonaEnergyModel,
    KonaPretrainConfig,
    batch_margin_loss,
)


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cross-domain Kona-style EBRM.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domains", nargs="+", default=DOMAIN_ORDER)
    parser.add_argument("--proof-state-path", default="data/proof_state_traces/run_attempt_only.jsonl")
    parser.add_argument("--checkpoint-out", default=None)
    return parser.parse_args()


def collate_examples(batch: list, tokenizer: KonaCharTokenizer) -> dict[str, torch.Tensor]:
    input_ids = []
    token_types = []
    attention_mask = []
    domain_ids = []
    labels = []
    for example in batch:
        ids, types, mask = tokenizer.encode_pair(example.context_text, example.state_text)
        input_ids.append(ids)
        token_types.append(types)
        attention_mask.append(mask)
        domain_ids.append(example.domain_id)
        labels.append(example.label)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "token_types": torch.tensor(token_types, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "domain_ids": torch.tensor(domain_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float32),
    }


@torch.no_grad()
def evaluate(model: KonaEnergyModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_ids = batch["domain_ids"].to(device)
        labels = batch["labels"].to(device)
        energy = model(input_ids, token_types, attention_mask, domain_ids)
        logits = -energy
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        preds = (logits >= 0).float()
        total_loss += loss.item() * input_ids.size(0)
        correct += int((preds == labels).sum().item())
        total += int(input_ids.size(0))
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = KonaCharTokenizer(max_length=args.max_seq_len)
    cfg = KonaPretrainConfig(
        num_domains=len(DOMAIN_ORDER),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    train_ds = KonaPretrainDataset(
        size=args.train_size,
        seed=args.seed,
        domains=args.domains,
        proof_state_path=args.proof_state_path,
    )
    val_ds = KonaPretrainDataset(
        size=args.val_size,
        seed=args.seed + 100_000,
        domains=args.domains,
        proof_state_path=args.proof_state_path,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda batch: collate_examples(batch, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: collate_examples(batch, tokenizer),
    )
    model = KonaEnergyModel(cfg, vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_iter = iter(train_loader)
    start = time.time()
    print("config", json.dumps(vars(args), indent=2))
    print("model", json.dumps(asdict(cfg), indent=2))
    print("device", device)

    for step in range(1, args.train_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        token_types = batch["token_types"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        domain_ids = batch["domain_ids"].to(device)
        labels = batch["labels"].to(device)
        energy = model(input_ids, token_types, attention_mask, domain_ids)
        logits = -energy
        loss_bce = F.binary_cross_entropy_with_logits(logits, labels)
        loss_margin = batch_margin_loss(energy, labels)
        loss = loss_bce + 0.5 * loss_margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 25 == 0 or step == 1:
            preds = (logits >= 0).float()
            acc = (preds == labels).float().mean().item()
            print(
                f"step={step} loss={loss.item():.4f} "
                f"bce={loss_bce.item():.4f} margin={loss_margin.item():.4f} "
                f"train_acc={acc:.4f}"
            )
        if step % args.eval_every == 0 or step == args.train_steps:
            metrics = evaluate(model, val_loader, device)
            print(
                f"eval step={step} "
                f"val_loss={metrics['loss']:.4f} "
                f"val_acc={metrics['accuracy']:.4f}"
            )

    elapsed = time.time() - start
    print(f"training_seconds={elapsed:.1f}")
    if args.checkpoint_out:
        payload = {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "tokenizer": {"max_length": tokenizer.max_length, "vocab_size": tokenizer.vocab_size},
            "domains": DOMAIN_ORDER,
        }
        checkpoint_path = Path(args.checkpoint_out)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)
        print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
