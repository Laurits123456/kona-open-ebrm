from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Protocol, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class KonaScoredState(Generic[StateT]):
    state: StateT
    energy: float
    solved: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KonaSearchStep:
    iteration: int
    frontier_size: int
    best_energy: float
    expanded_energy: float
    solved: bool


@dataclass(frozen=True)
class KonaSearchResult(Generic[StateT]):
    best: KonaScoredState[StateT]
    steps: list[KonaSearchStep]
    expansions: int
    elapsed_ms: float


class KonaDomain(Protocol[StateT]):
    def initial_states(self) -> list[StateT]:
        ...

    def state_key(self, state: StateT) -> Hashable:
        ...

    def score(self, state: StateT) -> KonaScoredState[StateT]:
        ...

    def expand(self, state: StateT) -> list[StateT]:
        ...


class KonaRuntime(Generic[StateT]):
    def __init__(
        self,
        *,
        beam_width: int = 16,
        max_expansions: int = 20_000,
        max_iterations: int = 20_000,
    ) -> None:
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.max_iterations = max_iterations

    def search(self, domain: KonaDomain[StateT]) -> KonaSearchResult[StateT]:
        started = time.time()
        counter = itertools.count()
        best_by_key: dict[Hashable, float] = {}
        queue: list[tuple[float, int, KonaScoredState[StateT]]] = []
        steps: list[KonaSearchStep] = []
        expansions = 0

        best_overall: KonaScoredState[StateT] | None = None
        for state in domain.initial_states():
            scored = domain.score(state)
            best_by_key[domain.state_key(scored.state)] = scored.energy
            heapq.heappush(queue, (scored.energy, next(counter), scored))
            if best_overall is None or scored.energy < best_overall.energy:
                best_overall = scored

        if best_overall is None:
            raise ValueError("Kona domain returned no initial states.")

        iteration = 0
        while queue and expansions < self.max_expansions and iteration < self.max_iterations:
            frontier_size = len(queue)
            current_batch: list[KonaScoredState[StateT]] = []
            for _ in range(min(self.beam_width, len(queue))):
                _, _, scored = heapq.heappop(queue)
                current_batch.append(scored)

            current_batch.sort(key=lambda item: item.energy)
            current = current_batch[0]
            if current.energy < best_overall.energy:
                best_overall = current

            steps.append(
                KonaSearchStep(
                    iteration=iteration,
                    frontier_size=frontier_size,
                    best_energy=best_overall.energy,
                    expanded_energy=current.energy,
                    solved=current.solved,
                )
            )
            if current.solved:
                break

            for scored in current_batch:
                if scored.solved:
                    continue
                for child in domain.expand(scored.state):
                    expansions += 1
                    child_scored = domain.score(child)
                    child_key = domain.state_key(child_scored.state)
                    previous_best = best_by_key.get(child_key)
                    if previous_best is not None and child_scored.energy >= previous_best:
                        if expansions >= self.max_expansions:
                            break
                        continue
                    best_by_key[child_key] = child_scored.energy
                    heapq.heappush(queue, (child_scored.energy, next(counter), child_scored))
                    if child_scored.energy < best_overall.energy:
                        best_overall = child_scored
                    if child_scored.solved:
                        steps.append(
                            KonaSearchStep(
                                iteration=iteration + 1,
                                frontier_size=len(queue),
                                best_energy=child_scored.energy,
                                expanded_energy=child_scored.energy,
                                solved=True,
                            )
                        )
                        elapsed_ms = (time.time() - started) * 1000.0
                        return KonaSearchResult(
                            best=child_scored,
                            steps=steps,
                            expansions=expansions,
                            elapsed_ms=elapsed_ms,
                        )
                    if expansions >= self.max_expansions:
                        break
                if expansions >= self.max_expansions:
                    break
            iteration += 1

        elapsed_ms = (time.time() - started) * 1000.0
        return KonaSearchResult(
            best=best_overall,
            steps=steps,
            expansions=expansions,
            elapsed_ms=elapsed_ms,
        )
