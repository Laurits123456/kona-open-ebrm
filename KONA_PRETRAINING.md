# Kona Cross-Domain Pretraining

This is the branch's first real step toward a generic learned EBRM instead of
task-local heuristics.

## What It Trains

`train_kona_ebrm.py` trains a single shared energy model over serialized
`(context, state)` pairs from multiple domains:

- graph shortest path
- arithmetic evaluation
- SAT assignment checking
- Sudoku completion
- proof-state recoverability

The model is in `kona/pretrain_model.py` and uses one shared character-level
Transformer encoder plus a scalar energy head.

## Why This Is Different

This is not "train one Sudoku scorer".

It is closer to small-scale language-model-style pretraining:

- shared vocabulary
- shared encoder
- mixed-domain batches
- same energy function across domains

## Command

```powershell
python train_kona_ebrm.py --train-steps 200 --batch-size 32 --checkpoint-out artifacts/kona_pretrain_smoke.pt
```
