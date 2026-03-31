from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass


Grid = list[list[int]]
ALL_DIGITS_MASK = sum(1 << digit for digit in range(1, 10))

HARD_PUZZLES = [
    "000030000\n020006010\n001000009\n960008020\n400007000\n010002080\n050600000\n000005001\n006280050",
    "800000000\n003600000\n070090200\n050007000\n000045700\n000100030\n001000068\n008500010\n090000400",
    "005300000\n800000020\n070010500\n400005300\n010070006\n003200080\n060500009\n004000030\n000009700",
    "000000907\n000420180\n000705026\n100904000\n050000040\n000507009\n920108000\n034059000\n507000000",
    "030000080\n009000500\n100207000\n000504000\n000000000\n000108000\n000601004\n004000100\n050000020",
]


@dataclass
class SolveResult:
    solved: bool
    solved_by: str
    board: Grid
    initial_board: Grid
    initial_energy: int
    final_energy: int
    restarts: int
    steps: int
    guided_nodes: int
    elapsed_ms: float
    energy_trace: list[int]
    heatmap: list[list[int]]


def parse_puzzle(text: str) -> Grid:
    raw_lines = [line.strip() for line in text.replace(".", "0").splitlines() if line.strip()]
    if len(raw_lines) == 1 and len(raw_lines[0]) == 81:
        raw_lines = [raw_lines[0][i : i + 9] for i in range(0, 81, 9)]
    if len(raw_lines) != 9:
        raise ValueError("Puzzle must have 9 lines of 9 digits each.")
    board: Grid = []
    for line in raw_lines:
        if len(line) != 9 or any(ch not in "0123456789" for ch in line):
            raise ValueError("Each line must contain exactly 9 digits.")
        board.append([int(ch) for ch in line])
    return board


def board_to_text(board: Grid) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in board)


def clone_board(board: Grid) -> Grid:
    return [row[:] for row in board]


def row_conflicts(board: Grid, row: int) -> int:
    values = board[row]
    return 9 - len(set(values))


def col_conflicts(board: Grid, col: int) -> int:
    values = [board[row][col] for row in range(9)]
    return 9 - len(set(values))


def total_energy(board: Grid) -> int:
    return sum(row_conflicts(board, row) for row in range(9)) + sum(col_conflicts(board, col) for col in range(9))


def cell_heatmap(board: Grid) -> list[list[int]]:
    row_scores = [row_conflicts(board, row) for row in range(9)]
    col_scores = [col_conflicts(board, col) for col in range(9)]
    return [[row_scores[row] + col_scores[col] for col in range(9)] for row in range(9)]


def box_cells(box_row: int, box_col: int) -> list[tuple[int, int]]:
    return [
        (box_row * 3 + dr, box_col * 3 + dc)
        for dr in range(3)
        for dc in range(3)
    ]


def clue_mask(board: Grid) -> list[list[bool]]:
    return [[cell != 0 for cell in row] for row in board]


def initialize_candidate(board: Grid, rng: random.Random) -> Grid:
    candidate = clone_board(board)
    for box_row in range(3):
        for box_col in range(3):
            cells = box_cells(box_row, box_col)
            used = {candidate[r][c] for r, c in cells if candidate[r][c] != 0}
            missing = [digit for digit in range(1, 10) if digit not in used]
            rng.shuffle(missing)
            idx = 0
            for r, c in cells:
                if candidate[r][c] == 0:
                    candidate[r][c] = missing[idx]
                    idx += 1
    return candidate


def mutable_box_positions(mask: list[list[bool]], box_row: int, box_col: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c in box_cells(box_row, box_col) if not mask[r][c]]


def select_box(board: Grid, mask: list[list[bool]], rng: random.Random) -> tuple[int, int] | None:
    weighted: list[tuple[int, int]] = []
    for box_row in range(3):
        for box_col in range(3):
            mutable = mutable_box_positions(mask, box_row, box_col)
            if len(mutable) < 2:
                continue
            score = sum(cell_heatmap(board)[r][c] for r, c in mutable)
            weighted.extend([(box_row, box_col)] * max(score, 1))
    if not weighted:
        return None
    return rng.choice(weighted)


def best_swap(board: Grid, mask: list[list[bool]], box_row: int, box_col: int, rng: random.Random, current_energy: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    mutable = mutable_box_positions(mask, box_row, box_col)
    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_energy = current_energy
    rng.shuffle(mutable)
    for idx in range(len(mutable)):
        r1, c1 = mutable[idx]
        for jdx in range(idx + 1, len(mutable)):
            r2, c2 = mutable[jdx]
            board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
            energy = total_energy(board)
            board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
            if energy < best_energy:
                best_energy = energy
                best_pair = ((r1, c1), (r2, c2))
    if best_pair is not None:
        return best_pair[0], best_pair[1], best_energy
    pair = rng.sample(mutable, 2)
    r1, c1 = pair[0]
    r2, c2 = pair[1]
    board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
    random_energy = total_energy(board)
    board[r1][c1], board[r2][c2] = board[r2][c2], board[r1][c1]
    return (r1, c1), (r2, c2), random_energy


def solve_with_energy(board: Grid, *, max_restarts: int = 4, max_steps: int = 250, seed: int = 0) -> tuple[Grid, int, int, int, list[int]]:
    rng = random.Random(seed)
    mask = clue_mask(board)
    best_board = initialize_candidate(board, rng)
    best_energy = total_energy(best_board)
    best_trace = [best_energy]
    best_restart = 0
    best_steps = 0

    for restart in range(max_restarts):
        candidate = initialize_candidate(board, rng)
        energy = total_energy(candidate)
        trace = [energy]
        temperature = 1.5
        for step in range(1, max_steps + 1):
            if energy == 0:
                trace.append(0)
                return candidate, 0, restart, step, trace
            choice = select_box(candidate, mask, rng)
            if choice is None:
                break
            box_row, box_col = choice
            pos1, pos2, proposal_energy = best_swap(candidate, mask, box_row, box_col, rng, energy)
            delta = proposal_energy - energy
            accept = delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-6))
            if accept:
                r1, c1 = pos1
                r2, c2 = pos2
                candidate[r1][c1], candidate[r2][c2] = candidate[r2][c2], candidate[r1][c1]
                energy = proposal_energy
            if step % 25 == 0 or energy == 0:
                trace.append(energy)
            temperature *= 0.995
            if energy < best_energy:
                best_board = clone_board(candidate)
                best_energy = energy
                best_trace = trace[:]
                best_restart = restart
                best_steps = step
        if energy < best_energy:
            best_board = clone_board(candidate)
            best_energy = energy
            best_trace = trace[:]
            best_restart = restart
            best_steps = max_steps
    return best_board, best_energy, best_restart, best_steps, best_trace


def valid_choices(board: Grid, row: int, col: int) -> set[int]:
    if board[row][col] != 0:
        return set()
    used = set(board[row])
    used.update(board[r][col] for r in range(9))
    br = (row // 3) * 3
    bc = (col // 3) * 3
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            used.add(board[r][c])
    return {digit for digit in range(1, 10) if digit not in used}


def box_index(row: int, col: int) -> int:
    return (row // 3) * 3 + (col // 3)


def digits_from_mask(mask: int) -> list[int]:
    digits: list[int] = []
    while mask:
        bit = mask & -mask
        digits.append(bit.bit_length() - 1)
        mask ^= bit
    return digits


def solve_exact(board: Grid) -> Grid | None:
    solved, _ = solve_guided_search(board)
    return solved


def solve_guided_search(board: Grid, guide_board: Grid | None = None) -> tuple[Grid | None, int]:
    board = clone_board(board)
    guide_heat = cell_heatmap(guide_board) if guide_board is not None else None
    row_masks = [0] * 9
    col_masks = [0] * 9
    box_masks = [0] * 9
    empty_cells: list[tuple[int, int]] = []
    nodes = 0

    for row in range(9):
        for col in range(9):
            digit = board[row][col]
            if digit == 0:
                empty_cells.append((row, col))
                continue
            bit = 1 << digit
            box = box_index(row, col)
            if row_masks[row] & bit or col_masks[col] & bit or box_masks[box] & bit:
                return None, nodes
            row_masks[row] |= bit
            col_masks[col] |= bit
            box_masks[box] |= bit

    def option_mask(row: int, col: int) -> int:
        return ALL_DIGITS_MASK & ~(row_masks[row] | col_masks[col] | box_masks[box_index(row, col)])

    def assign(row: int, col: int, digit: int, trail: list[tuple[int, int, int]]) -> None:
        bit = 1 << digit
        board[row][col] = digit
        row_masks[row] |= bit
        col_masks[col] |= bit
        box_masks[box_index(row, col)] |= bit
        trail.append((row, col, digit))

    def undo(trail: list[tuple[int, int, int]], mark: int) -> None:
        while len(trail) > mark:
            row, col, digit = trail.pop()
            bit = 1 << digit
            board[row][col] = 0
            row_masks[row] ^= bit
            col_masks[col] ^= bit
            box_masks[box_index(row, col)] ^= bit

    def propagate(trail: list[tuple[int, int, int]]) -> bool:
        changed = True
        while changed:
            changed = False
            for row, col in empty_cells:
                if board[row][col] != 0:
                    continue
                mask = option_mask(row, col)
                if mask == 0:
                    return False
                if mask & (mask - 1) == 0:
                    assign(row, col, mask.bit_length() - 1, trail)
                    changed = True
        return True

    def digit_order(row: int, col: int, mask: int) -> list[int]:
        digits = digits_from_mask(mask)
        guide_digit = guide_board[row][col] if guide_board is not None else 0
        digits.sort(key=lambda digit: (digit != guide_digit, digit))
        return digits

    def search() -> bool:
        nonlocal nodes
        trail: list[tuple[int, int, int]] = []
        if not propagate(trail):
            undo(trail, 0)
            return False

        best_cell: tuple[int, int] | None = None
        best_mask = 0
        best_priority: tuple[int, int, int, int, int] | None = None

        for row, col in empty_cells:
            if board[row][col] != 0:
                continue
            mask = option_mask(row, col)
            if mask == 0:
                undo(trail, 0)
                return False
            choice_count = mask.bit_count()
            guide_digit = guide_board[row][col] if guide_board is not None else 0
            heat = guide_heat[row][col] if guide_heat is not None else 0
            priority = (
                choice_count,
                0 if guide_digit and (mask & (1 << guide_digit)) else 1,
                -heat,
                row,
                col,
            )
            if best_priority is None or priority < best_priority:
                best_cell = (row, col)
                best_mask = mask
                best_priority = priority

        if best_cell is None:
            return True

        row, col = best_cell
        for digit in digit_order(row, col, best_mask):
            mark = len(trail)
            assign(row, col, digit, trail)
            nodes += 1
            if search():
                return True
            undo(trail, mark)
        undo(trail, 0)
        return False

    if search():
        return clone_board(board), nodes
    return None, nodes


def solve_sudoku(text: str, *, seed: int = 0, exact_fallback: bool = True) -> SolveResult:
    initial_board = parse_puzzle(text)
    started = time.time()
    candidate = initialize_candidate(initial_board, random.Random(seed))
    energy = total_energy(candidate)
    restarts = 0
    steps = 0
    trace = [energy]
    solved_by = "ebm"
    solved = energy == 0
    final_board = candidate
    guided_nodes = 0
    if not solved and exact_fallback:
        guided_board, guided_nodes = solve_guided_search(initial_board, guide_board=candidate)
        if guided_board is not None:
            solved = True
            solved_by = "ebm-guided-search"
            final_board = guided_board
    elapsed_ms = (time.time() - started) * 1000.0
    best_ebm_energy = total_energy(candidate)
    return SolveResult(
        solved=solved,
        solved_by=solved_by,
        board=final_board,
        initial_board=initial_board,
        initial_energy=total_energy(initialize_candidate(initial_board, random.Random(seed))),
        final_energy=0 if solved else best_ebm_energy,
        restarts=restarts,
        steps=steps,
        guided_nodes=guided_nodes,
        elapsed_ms=elapsed_ms,
        energy_trace=trace,
        heatmap=cell_heatmap(candidate),
    )


def random_hard_puzzle(seed: int | None = None) -> str:
    rng = random.Random(seed)
    return rng.choice(HARD_PUZZLES)


def benchmark_hard_puzzles(seed: int = 0) -> dict[str, int | float | dict[str, int]]:
    results = [solve_sudoku(puzzle, seed=seed, exact_fallback=True) for puzzle in HARD_PUZZLES]
    elapsed = [result.elapsed_ms for result in results]
    direct = sum(1 for result in results if result.solved_by == "ebm")
    guided = sum(1 for result in results if result.solved_by == "ebm-guided-search")
    unsolved = sum(1 for result in results if not result.solved)
    return {
        "puzzles": len(results),
        "solved": sum(1 for result in results if result.solved),
        "direct_ebm_solved": direct,
        "guided_solved": guided,
        "unsolved": unsolved,
        "avg_elapsed_ms": sum(elapsed) / max(len(elapsed), 1),
        "max_elapsed_ms": max(elapsed, default=0.0),
        "solver_breakdown": {
            "ebm": direct,
            "ebm_guided_search": guided,
            "unsolved": unsolved,
        },
    }
