# kona-open-ebrm

Open cross-domain energy-based reasoning model.

This repo is a standalone export of the generic EBRM work from the larger
`EBRMS` project. It contains:

- a reusable energy-search runtime
- a mixed-domain pretraining pipeline
- a generic pair scorer
- a Sudoku demo app

## Included domains

- graph shortest path
- arithmetic evaluation
- SAT assignment checking
- Sudoku board plausibility
- hooks for broader cross-domain extensions

## What this is

- real learned energy model training code
- generic across multiple domains
- suitable as a base for broader EBRM pretraining

## What this is not yet

- a chat model
- a frontier-scale pretrained reasoner
- a full Putnam theorem prover by itself

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train_kona_ebrm.py --train-steps 200 --checkpoint-out artifacts/kona_pretrain_smoke.pt
```

## Score a pair

```powershell
python kona_cli.py score-pair --checkpoint artifacts/kona_pretrain_smoke.pt --domain arithmetic --context "arithmetic`nexpr=2 + 3 * 4" --state "answer=14"
```

## Run the Sudoku demo

```powershell
python sudoku_demo_app.py
```

Or on Windows:

```powershell
.\start_sudoku_demo.ps1
```
