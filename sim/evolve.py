"""Population, selection, and the five evolutionary regimes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.agent import (
    Genome,
    apply_controller,
    apply_harness,
    apply_mutation,
    canonical_harness,
    random_genome,
)
from sim.config import RunConfig
from sim.tasks import evaluate_genome, make_generation_batches


@dataclass
class GenerationLog:
    elite_fitness: float
    mean_fitness: float
    best_fitness: float
    elite_deceptive: float
    elite_n_tools: float
    elite_key_rate: float
    elite_bait_rate: float
    tool_turnover: float
    elite_tool_sets: list[frozenset[int]] = field(default_factory=list)


def _elitism(regime: str, cfg: RunConfig) -> int:
    if regime == "coevolution_conservative":
        return cfg.conservative_elites
    if regime == "coevolution":
        return 1
    return cfg.n_elites


def _crossover(
    a: Genome, b: Genome, rng: np.random.Generator, regime: str
) -> Genome:
    child = a.copy()
    if rng.random() < 0.5:
        # Mix controller slots.
        k = child.prototypes.shape[0]
        take = rng.random(k) < 0.5
        child.prototypes[take] = b.prototypes[take]
        child.slot_tool[take] = b.slot_tool[take]
    if regime != "model_first" and rng.random() < 0.5:
        mix = rng.random(child.tool_on.size) < 0.5
        child.tool_on[mix] = b.tool_on[mix]
        child.tool_noise[mix] = b.tool_noise[mix]
        smix = rng.random(child.sensor_on.size) < 0.5
        child.sensor_on[smix] = b.sensor_on[smix]
        child.sensor_noise[smix] = b.sensor_noise[smix]
    return child


def _turnover(prev: list[frozenset[int]], cur: list[frozenset[int]]) -> float:
    if not prev:
        return 0.0
    # Mean Jaccard distance between matched elite ranks.
    n = min(len(prev), len(cur))
    dists = []
    for i in range(n):
        u = prev[i] | cur[i]
        if not u:
            dists.append(0.0)
            continue
        dists.append(1.0 - len(prev[i] & cur[i]) / len(u))
    return float(np.mean(dists)) if dists else 0.0


def _init_population(
    rng: np.random.Generator, cfg: RunConfig, regime: str, capacity: int
) -> tuple[list[Genome], dict | None, Genome | None]:
    frozen_h = canonical_harness(cfg) if regime == "model_first" else None
    frozen_c = None
    if regime in ("harness_first", "brain_directed"):
        frozen_c = random_genome(rng, cfg, capacity)
    pop = []
    for _ in range(cfg.pop_size):
        g = random_genome(rng, cfg, capacity)
        if frozen_h is not None:
            apply_harness(g, frozen_h)
        if frozen_c is not None:
            apply_controller(g, frozen_c)
        pop.append(g)
    return pop, frozen_h, frozen_c


def _stamp_frozen(g: Genome, frozen_h: dict | None, frozen_c: Genome | None) -> None:
    if frozen_h is not None:
        apply_harness(g, frozen_h)
    if frozen_c is not None:
        apply_controller(g, frozen_c)


def run_evolution(
    rng: np.random.Generator,
    cfg: RunConfig,
    regime: str,
    capacity: int,
) -> tuple[list[GenerationLog], dict]:
    pop, frozen_h, frozen_c = _init_population(rng, cfg, regime, capacity)
    logs: list[GenerationLog] = []
    prev_elite_sets: list[frozenset[int]] = []
    last_stats: list[dict] = []

    for _gen in range(cfg.n_generations):
        batches = make_generation_batches(rng, cfg)
        stats = [evaluate_genome(g, batches, cfg) for g in pop]
        last_stats = stats
        # True fitness is always logged. Brain-directed *selects* on the
        # myopic score so a greedy editor can lock in a locally good limb.
        select_key = "myopic_fitness" if regime == "brain_directed" else "fitness"
        fit = np.array([s["fitness"] for s in stats], dtype=np.float64)
        select = np.array([s[select_key] for s in stats], dtype=np.float64)
        order = np.argsort(-select)
        n_elite = _elitism(regime, cfg)
        elite_idx = order[: max(n_elite, cfg.n_parents)]
        elite_view = order[: max(8, n_elite)]
        elite_sets = [stats[i]["tool_set"] for i in elite_view]
        log = GenerationLog(
            elite_fitness=float(fit[elite_view].mean()),
            mean_fitness=float(fit.mean()),
            best_fitness=float(fit[order[0]]),
            elite_deceptive=float(np.mean([stats[i]["deceptive"] for i in elite_view])),
            elite_n_tools=float(np.mean([stats[i]["n_tools"] for i in elite_view])),
            elite_key_rate=float(np.mean([stats[i]["has_key"] for i in elite_view])),
            elite_bait_rate=float(np.mean([stats[i]["has_bait"] for i in elite_view])),
            tool_turnover=_turnover(prev_elite_sets, elite_sets),
            elite_tool_sets=elite_sets,
        )
        logs.append(log)
        prev_elite_sets = elite_sets

        parents = [pop[i] for i in elite_idx]
        parent_stats = [stats[i] for i in elite_idx]
        next_pop: list[Genome] = [pop[i].copy() for i in order[:n_elite]]
        while len(next_pop) < cfg.pop_size:
            i = int(rng.integers(0, len(parents)))
            j = int(rng.integers(0, len(parents)))
            child = _crossover(parents[i], parents[j], rng, regime)
            apparent = parent_stats[i]["apparent"]
            apply_mutation(child, rng, regime, cfg, apparent)
            _stamp_frozen(child, frozen_h, frozen_c)
            next_pop.append(child)
        pop = next_pop

    final = last_stats
    order = np.argsort([-s["fitness"] for s in final])
    best = final[int(order[0])]
    summary = {
        "final_elite_fitness": logs[-1].elite_fitness,
        "final_best_fitness": logs[-1].best_fitness,
        "final_mean_fitness": logs[-1].mean_fitness,
        "final_deceptive": logs[-1].elite_deceptive,
        "final_key_rate": logs[-1].elite_key_rate,
        "final_bait_rate": logs[-1].elite_bait_rate,
        "final_n_tools": logs[-1].elite_n_tools,
        "mean_turnover": float(np.mean([lg.tool_turnover for lg in logs[1:]])),
        "late_turnover": float(np.mean([lg.tool_turnover for lg in logs[len(logs) // 2 :]])),
        "best_has_key": bool(best["has_key"]),
        "best_n_tools": int(best["n_tools"]),
        "best_n_sensors": int(best["n_sensors"]),
        "best_forage": float(best["forage"]),
        "best_bandit": float(best["bandit"]),
        "best_deceptive": float(best["deceptive"]),
    }
    return logs, summary
