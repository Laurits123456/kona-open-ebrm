from __future__ import annotations

import argparse

from kona.pretrain_runtime import KonaPretrainScorer
from kona.sudoku import HARD_PUZZLES, benchmark_hard_puzzles, solve_sudoku


def run_sudoku(args: argparse.Namespace) -> int:
    if args.benchmark:
        print(benchmark_hard_puzzles(seed=args.seed))
        return 0
    puzzle = args.puzzle or HARD_PUZZLES[0]
    result = solve_sudoku(puzzle, seed=args.seed)
    print(
        {
            "solved": result.solved,
            "solver": result.solved_by,
            "elapsed_ms": round(result.elapsed_ms, 3),
            "guided_nodes": result.guided_nodes,
            "initial_energy": result.initial_energy,
            "final_energy": result.final_energy,
        }
    )
    return 0


def run_score_pair(args: argparse.Namespace) -> int:
    scorer = KonaPretrainScorer.from_checkpoint(args.checkpoint, device=args.device)
    score = scorer.score_pair(args.domain, args.context, args.state)
    print(
        {
            "domain": score.domain_name,
            "energy": round(score.energy, 6),
            "plausibility_prob": round(score.plausibility_prob, 6),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open Kona-style EBRM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sudoku = subparsers.add_parser("sudoku")
    sudoku.add_argument("--puzzle", default=None)
    sudoku.add_argument("--seed", type=int, default=0)
    sudoku.add_argument("--benchmark", action="store_true")
    sudoku.set_defaults(func=run_sudoku)

    pair = subparsers.add_parser("score-pair")
    pair.add_argument("--checkpoint", required=True)
    pair.add_argument("--domain", required=True, choices=["graph", "arithmetic", "sat", "sudoku", "proof_state"])
    pair.add_argument("--device", default="cpu")
    pair.add_argument("--context", required=True)
    pair.add_argument("--state", required=True)
    pair.set_defaults(func=run_score_pair)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
