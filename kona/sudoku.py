from __future__ import annotations

import random
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache

from kona.core import KonaDomain, KonaRuntime, KonaScoredState
from sudoku_ebm import (
    HARD_PUZZLES,
    Grid,
    SolveResult,
    board_to_text,
    cell_heatmap,
    clone_board,
    initialize_candidate,
    parse_puzzle,
    random_hard_puzzle,
    total_energy,
)


ALL_DIGITS_MASK = sum(1 << digit for digit in range(1, 10))


@dataclass(frozen=True)
class SudokuAnalysis:
    solved: bool
    invalid_penalty: int
    empties: int
    branching_penalty: int
    best_index: int
    best_mask: int


class KonaSudokuDomain(KonaDomain[tuple[int, ...]]):
    def __init__(self, puzzle: Grid, guide_board: Grid) -> None:
        self.puzzle = tuple(cell for row in puzzle for cell in row)
        self.guide = tuple(cell for row in guide_board for cell in row)

    def initial_states(self) -> list[tuple[int, ...]]:
        return [self.puzzle]

    def state_key(self, state: tuple[int, ...]) -> Hashable:
        return state

    def score(self, state: tuple[int, ...]) -> KonaScoredState[tuple[int, ...]]:
        analysis = _analyze_state(state, self.guide)
        guide_mismatch = sum(
            1
            for idx, digit in enumerate(state)
            if digit != 0 and self.puzzle[idx] == 0 and self.guide[idx] != 0 and digit != self.guide[idx]
        )
        energy = (
            float(analysis.invalid_penalty * 10_000)
            + float(analysis.empties * 100)
            + float(analysis.branching_penalty)
            + float(guide_mismatch) * 0.25
        )
        metadata = {
            "empties": analysis.empties,
            "branching_penalty": analysis.branching_penalty,
            "guide_mismatch": guide_mismatch,
            "invalid_penalty": analysis.invalid_penalty,
        }
        return KonaScoredState(
            state=state,
            energy=energy,
            solved=analysis.solved,
            metadata=metadata,
        )

    def expand(self, state: tuple[int, ...]) -> list[tuple[int, ...]]:
        analysis = _analyze_state(state, self.guide)
        if analysis.solved or analysis.best_index < 0 or analysis.best_mask == 0:
            return []
        digits = _digits_from_mask(analysis.best_mask)
        guide_digit = self.guide[analysis.best_index]
        digits.sort(key=lambda digit: (digit != guide_digit, digit))
        next_states: list[tuple[int, ...]] = []
        for digit in digits:
            child = list(state)
            child[analysis.best_index] = digit
            next_states.append(tuple(child))
        return next_states


def _digits_from_mask(mask: int) -> list[int]:
    digits: list[int] = []
    while mask:
        bit = mask & -mask
        digits.append(bit.bit_length() - 1)
        mask ^= bit
    return digits


@lru_cache(maxsize=200_000)
def _analyze_state(state: tuple[int, ...], guide: tuple[int, ...]) -> SudokuAnalysis:
    row_masks = [0] * 9
    col_masks = [0] * 9
    box_masks = [0] * 9
    invalid_penalty = 0
    empties = 0

    for idx, digit in enumerate(state):
        if digit == 0:
            empties += 1
            continue
        row = idx // 9
        col = idx % 9
        box = (row // 3) * 3 + (col // 3)
        bit = 1 << digit
        if row_masks[row] & bit:
            invalid_penalty += 1
        else:
            row_masks[row] |= bit
        if col_masks[col] & bit:
            invalid_penalty += 1
        else:
            col_masks[col] |= bit
        if box_masks[box] & bit:
            invalid_penalty += 1
        else:
            box_masks[box] |= bit

    if invalid_penalty:
        return SudokuAnalysis(
            solved=False,
            invalid_penalty=invalid_penalty,
            empties=empties,
            branching_penalty=empties * 9,
            best_index=-1,
            best_mask=0,
        )

    best_index = -1
    best_mask = 0
    best_count = 10
    branching_penalty = 0
    for idx, digit in enumerate(state):
        if digit != 0:
            continue
        row = idx // 9
        col = idx % 9
        box = (row // 3) * 3 + (col // 3)
        mask = ALL_DIGITS_MASK & ~(row_masks[row] | col_masks[col] | box_masks[box])
        count = mask.bit_count()
        if count == 0:
            return SudokuAnalysis(
                solved=False,
                invalid_penalty=1,
                empties=empties,
                branching_penalty=empties * 9,
                best_index=idx,
                best_mask=0,
            )
        branching_penalty += count - 1
        guide_digit = guide[idx]
        if count < best_count or (count == best_count and guide_digit != 0 and mask & (1 << guide_digit)):
            best_count = count
            best_index = idx
            best_mask = mask

    return SudokuAnalysis(
        solved=empties == 0,
        invalid_penalty=0,
        empties=empties,
        branching_penalty=branching_penalty,
        best_index=best_index,
        best_mask=best_mask,
    )


def _flat_to_board(state: tuple[int, ...]) -> Grid:
    return [list(state[row * 9 : (row + 1) * 9]) for row in range(9)]


def _build_guide_board(board: Grid, seed: int, samples: int = 4) -> tuple[Grid, int]:
    rng = random.Random(seed)
    best_board: Grid | None = None
    best_energy: int | None = None
    for _ in range(samples):
        candidate = initialize_candidate(board, rng)
        energy = total_energy(candidate)
        if best_board is None or best_energy is None or energy < best_energy:
            best_board = clone_board(candidate)
            best_energy = energy
    if best_board is None or best_energy is None:
        raise ValueError("Failed to build Sudoku guide board.")
    return best_board, best_energy


def solve_sudoku(text: str, *, seed: int = 0, exact_fallback: bool = True) -> SolveResult:
    del exact_fallback
    initial_board = parse_puzzle(text)
    guide_board, guide_energy = _build_guide_board(initial_board, seed)
    runtime = KonaRuntime[tuple[int, ...]](beam_width=12, max_expansions=50_000, max_iterations=50_000)
    result = runtime.search(KonaSudokuDomain(initial_board, guide_board))
    final_board = _flat_to_board(result.best.state)
    trace = [int(step.best_energy) for step in result.steps] or [guide_energy]
    return SolveResult(
        solved=result.best.solved,
        solved_by="kona-best-first" if result.best.solved else "kona-partial",
        board=final_board,
        initial_board=initial_board,
        initial_energy=guide_energy,
        final_energy=0 if result.best.solved else int(result.best.energy),
        restarts=0,
        steps=len(result.steps),
        guided_nodes=result.expansions,
        elapsed_ms=result.elapsed_ms,
        energy_trace=trace,
        heatmap=cell_heatmap(guide_board),
    )


def benchmark_hard_puzzles(seed: int = 0) -> dict[str, int | float | dict[str, int]]:
    results = [solve_sudoku(puzzle, seed=seed) for puzzle in HARD_PUZZLES]
    elapsed = [result.elapsed_ms for result in results]
    direct = sum(1 for result in results if result.solved_by == "kona-best-first")
    unsolved = sum(1 for result in results if not result.solved)
    return {
        "puzzles": len(results),
        "solved": sum(1 for result in results if result.solved),
        "direct_ebm_solved": direct,
        "guided_solved": 0,
        "unsolved": unsolved,
        "avg_elapsed_ms": sum(elapsed) / max(len(elapsed), 1),
        "max_elapsed_ms": max(elapsed, default=0.0),
        "solver_breakdown": {
            "kona_best_first": direct,
            "unsolved": unsolved,
        },
    }


__all__ = [
    "SolveResult",
    "benchmark_hard_puzzles",
    "board_to_text",
    "parse_puzzle",
    "random_hard_puzzle",
    "solve_sudoku",
]
