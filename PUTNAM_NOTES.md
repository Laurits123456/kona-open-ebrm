# Putnam Notes

This standalone repo is the open cross-domain EBRM export.

## Important Scope Caveat

The official staged PutnamBench result currently comes from the larger parent
project's proof-state EBRM integration, not from the standalone generic
`kona-open-ebrm` checkpoint alone.

## Current Observed Scores In The Parent Project

- Best observed staged run in this branch: `316 / 600`
  - `trusted_core = 300 / 300`
  - `untrusted_core = 16 / 300`
- Latest rerun while exporting this repo: `100 / 300`
  - `trusted_core = 100 / 300`

The live benchmark is provider-backed and not deterministic, so the score can
move materially between reruns.

## Interpretation

The standalone repo is a reusable generic EBRM base.

The parent project still contains the stronger theorem-specific proof-state
checkpoint and runtime logic needed for the current PutnamBench results.
