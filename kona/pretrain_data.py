from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


DOMAIN_ORDER = ["graph", "arithmetic", "sat", "sudoku", "proof_state"]
DOMAIN_IDS = {name: idx for idx, name in enumerate(DOMAIN_ORDER)}

BASE_SUDOKU = (
    (1, 2, 3, 4, 5, 6, 7, 8, 9),
    (4, 5, 6, 7, 8, 9, 1, 2, 3),
    (7, 8, 9, 1, 2, 3, 4, 5, 6),
    (2, 3, 4, 5, 6, 7, 8, 9, 1),
    (5, 6, 7, 8, 9, 1, 2, 3, 4),
    (8, 9, 1, 2, 3, 4, 5, 6, 7),
    (3, 4, 5, 6, 7, 8, 9, 1, 2),
    (6, 7, 8, 9, 1, 2, 3, 4, 5),
    (9, 1, 2, 3, 4, 5, 6, 7, 8),
)


@dataclass(frozen=True)
class KonaPretrainExample:
    domain_name: str
    domain_id: int
    context_text: str
    state_text: str
    label: int


def _dijkstra(adjacency: list[list[int]], source: int, sink: int) -> list[int]:
    import heapq

    dist = [math.inf] * len(adjacency)
    parent = [-1] * len(adjacency)
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist[node]:
            continue
        if node == sink:
            break
        for nxt, weight in enumerate(adjacency[node]):
            if weight <= 0:
                continue
            cand = cur_dist + weight
            if cand < dist[nxt]:
                dist[nxt] = cand
                parent[nxt] = node
                heapq.heappush(heap, (cand, nxt))
    if not math.isfinite(dist[sink]):
        return []
    path: list[int] = []
    cur = sink
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path if path and path[0] == source else []


def _serialize_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _generate_graph_example(rng: random.Random, label: int) -> KonaPretrainExample:
    n_nodes = 8
    while True:
        adjacency = [[0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and rng.random() < 0.28:
                    adjacency[i][j] = rng.randint(1, 9)
        source = rng.randrange(n_nodes)
        sink = rng.randrange(n_nodes)
        if source == sink:
            continue
        path = _dijkstra(adjacency, source, sink)
        if path:
            break
    context = "graph\n" + "\n".join(",".join(str(weight) for weight in row) for row in adjacency)
    context += f"\nsource={source}\nsink={sink}"
    if label:
        candidate = path[:]
    else:
        candidate = path[:]
        if rng.random() < 0.5 and len(candidate) > 1:
            candidate = candidate[:-1]
        else:
            replace_idx = rng.randrange(len(candidate))
            candidate[replace_idx] = rng.randrange(n_nodes)
    state = "path=" + ",".join(str(node) for node in candidate)
    return KonaPretrainExample("graph", DOMAIN_IDS["graph"], context, state, label)


def _generate_arithmetic_example(rng: random.Random, label: int) -> KonaPretrainExample:
    n_terms = rng.randint(2, 5)
    numbers = [str(rng.randint(0, 9)) for _ in range(n_terms)]
    operators = [rng.choice(["+", "-", "*"]) for _ in range(n_terms - 1)]
    tokens: list[str] = []
    for idx, number in enumerate(numbers):
        tokens.append(number)
        if idx < len(operators):
            tokens.append(operators[idx])

    stack: list[int] = []
    ops: list[str] = []
    prec = {"+": 1, "-": 1, "*": 2}

    def apply_top() -> None:
        b = stack.pop()
        a = stack.pop()
        op = ops.pop()
        if op == "+":
            stack.append(a + b)
        elif op == "-":
            stack.append(a - b)
        else:
            stack.append(a * b)

    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
        else:
            while ops and prec.get(ops[-1], 0) >= prec[token]:
                apply_top()
            ops.append(token)
    while ops:
        apply_top()
    value = stack[0]
    answer = value if label else value + rng.choice([-7, -5, -3, -2, 2, 3, 5, 7])
    context = "arithmetic\nexpr=" + " ".join(tokens)
    state = f"answer={answer}"
    return KonaPretrainExample("arithmetic", DOMAIN_IDS["arithmetic"], context, state, label)


def _generate_sat_example(rng: random.Random, label: int) -> KonaPretrainExample:
    n_vars = 5
    n_clauses = 6
    assignment = [rng.randint(0, 1) for _ in range(n_vars)]
    clauses: list[list[str]] = []
    for _ in range(n_clauses):
        vars_idx = rng.sample(range(n_vars), 3)
        satisfied_position = rng.randrange(3)
        clause: list[str] = []
        for pos, var_idx in enumerate(vars_idx):
            sign = assignment[var_idx] if pos == satisfied_position else rng.randint(0, 1)
            prefix = "" if sign else "!"
            clause.append(f"{prefix}x{var_idx}")
        clauses.append(clause)
    candidate = assignment[:]
    if not label:
        flip_count = rng.randint(1, 2)
        for idx in rng.sample(range(n_vars), flip_count):
            candidate[idx] = 1 - candidate[idx]
    context = "sat\n" + "\n".join(" ".join(clause) for clause in clauses)
    state = "assignment=" + ",".join(str(bit) for bit in candidate)
    return KonaPretrainExample("sat", DOMAIN_IDS["sat"], context, state, label)


def _permute_sudoku(rng: random.Random) -> list[list[int]]:
    board = [list(row) for row in BASE_SUDOKU]
    digit_perm = list(range(1, 10))
    rng.shuffle(digit_perm)
    digit_map = {idx + 1: digit_perm[idx] for idx in range(9)}
    board = [[digit_map[cell] for cell in row] for row in board]

    def permute_groups(values: list[list[int]]) -> list[list[int]]:
        groups = [values[idx : idx + 3] for idx in range(0, 9, 3)]
        rng.shuffle(groups)
        out: list[list[int]] = []
        for group in groups:
            rng.shuffle(group)
            out.extend(group)
        return out

    board = permute_groups(board)
    cols = permute_groups([list(col) for col in zip(*board, strict=False)])
    return [list(row) for row in zip(*cols, strict=False)]


def _mask_sudoku(board: list[list[int]], rng: random.Random, clues: int) -> list[list[int]]:
    puzzle = [row[:] for row in board]
    indexes = list(range(81))
    rng.shuffle(indexes)
    for idx in indexes[: 81 - clues]:
        puzzle[idx // 9][idx % 9] = 0
    return puzzle


def _corrupt_sudoku(board: list[list[int]], puzzle: list[list[int]], rng: random.Random) -> list[list[int]]:
    corrupted = [row[:] for row in board]
    mutable = [idx for idx in range(81) if puzzle[idx // 9][idx % 9] == 0]
    if not mutable:
        mutable = list(range(81))
    idx = rng.choice(mutable)
    row = idx // 9
    col = idx % 9
    choices = [digit for digit in range(1, 10) if digit != corrupted[row][col]]
    corrupted[row][col] = rng.choice(choices)
    return corrupted


def _generate_sudoku_example(rng: random.Random, label: int) -> KonaPretrainExample:
    solution = _permute_sudoku(rng)
    puzzle = _mask_sudoku(solution, rng, clues=rng.randint(28, 38))
    state_board = solution if label else _corrupt_sudoku(solution, puzzle, rng)
    context = "sudoku\npuzzle=\n" + _serialize_grid(puzzle)
    state = "board=\n" + _serialize_grid(state_board)
    return KonaPretrainExample("sudoku", DOMAIN_IDS["sudoku"], context, state, label)


def _normalize_label(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "recoverable"}:
            return 1
        if lowered in {"0", "false", "no", "unrecoverable"}:
            return 0
    return None


def load_proof_state_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = _normalize_label(row.get("recoverable_label"))
            if label is None:
                continue
            row["recoverable_label"] = label
            rows.append(row)
    return rows


def _serialize_proof_state_row(row: dict) -> tuple[str, str]:
    theorem = str(row.get("theorem_name") or row.get("problem_name") or "unknown")
    source_kind = str(row.get("source_state_kind") or "unknown")
    checker_feedback = str(row.get("checker_feedback") or "")[:240]
    context = (
        "proof_state\n"
        f"theorem={theorem}\n"
        f"source={source_kind}\n"
        f"feedback={checker_feedback}"
    )
    candidate = str(row.get("candidate_proof") or row.get("response_content") or "")[:360]
    state = "candidate=\n" + candidate
    return context, state


class KonaPretrainDataset(Dataset[KonaPretrainExample]):
    def __init__(
        self,
        *,
        size: int,
        seed: int = 0,
        domains: list[str] | None = None,
        proof_state_path: str | Path | None = None,
    ) -> None:
        requested = domains[:] if domains else DOMAIN_ORDER[:]
        self.proof_state_rows = (
            load_proof_state_rows(proof_state_path)
            if proof_state_path and Path(proof_state_path).exists()
            else []
        )
        self.domains = [
            domain
            for domain in requested
            if domain != "proof_state" or self.proof_state_rows
        ]
        if not self.domains:
            raise ValueError("No valid domains available for Kona pretraining.")
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> KonaPretrainExample:
        rng = random.Random(self.seed + index)
        domain = self.domains[index % len(self.domains)]
        label = rng.randint(0, 1)
        if domain == "graph":
            return _generate_graph_example(rng, label)
        if domain == "arithmetic":
            return _generate_arithmetic_example(rng, label)
        if domain == "sat":
            return _generate_sat_example(rng, label)
        if domain == "sudoku":
            return _generate_sudoku_example(rng, label)

        row = self.proof_state_rows[(self.seed + index) % len(self.proof_state_rows)]
        context, state = _serialize_proof_state_row(row)
        return KonaPretrainExample(
            "proof_state",
            DOMAIN_IDS["proof_state"],
            context,
            state,
            int(row["recoverable_label"]),
        )
