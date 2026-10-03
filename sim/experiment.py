"""Run the full comparison: regimes × capacities × seeds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from sim.config import REGIME_NAMES, RunConfig
from sim.evolve import GenerationLog, run_evolution


def _seed_rng(base_seed: int, regime: str, capacity: int) -> np.random.Generator:
    rid = REGIME_NAMES.index(regime) if regime in REGIME_NAMES else 17
    ss = np.random.SeedSequence([int(base_seed), rid, int(capacity), 0xA11CE])
    return np.random.default_rng(ss)


def _series(logs: list[GenerationLog], attr: str) -> list[float]:
    return [float(getattr(lg, attr)) for lg in logs]


def run_cell(
    cfg: RunConfig, regime: str, capacity: int, seed: int
) -> dict:
    rng = _seed_rng(seed, regime, capacity)
    logs, summary = run_evolution(rng, cfg, regime, capacity)
    return {
        "regime": regime,
        "capacity": capacity,
        "seed": seed,
        "elite_fitness": _series(logs, "elite_fitness"),
        "mean_fitness": _series(logs, "mean_fitness"),
        "best_fitness": _series(logs, "best_fitness"),
        "elite_deceptive": _series(logs, "elite_deceptive"),
        "elite_n_tools": _series(logs, "elite_n_tools"),
        "elite_key_rate": _series(logs, "elite_key_rate"),
        "elite_bait_rate": _series(logs, "elite_bait_rate"),
        "tool_turnover": _series(logs, "tool_turnover"),
        **summary,
    }


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
        "values": [float(v) for v in arr],
    }


def _stack_mean(cells: list[dict], key: str) -> tuple[list[float], list[float]]:
    mat = np.array([c[key] for c in cells], dtype=np.float64)
    return mat.mean(axis=0).tolist(), mat.std(axis=0, ddof=1).tolist()


def aggregate(cells: list[dict], cfg: RunConfig) -> dict:
    by_rc: dict[tuple[str, int], list[dict]] = {}
    for c in cells:
        by_rc.setdefault((c["regime"], c["capacity"]), []).append(c)

    grouped = {}
    for (regime, cap), group in sorted(by_rc.items()):
        fit_m, fit_s = _stack_mean(group, "elite_fitness")
        turn_m, turn_s = _stack_mean(group, "tool_turnover")
        key_m, key_s = _stack_mean(group, "elite_key_rate")
        dec_m, dec_s = _stack_mean(group, "elite_deceptive")
        grouped[f"{regime}|k={cap}"] = {
            "regime": regime,
            "capacity": cap,
            "n_seeds": len(group),
            "elite_fitness_mean": fit_m,
            "elite_fitness_std": fit_s,
            "turnover_mean": turn_m,
            "turnover_std": turn_s,
            "key_rate_mean": key_m,
            "key_rate_std": key_s,
            "deceptive_mean": dec_m,
            "deceptive_std": dec_s,
            "final_elite_fitness": _mean_std([g["final_elite_fitness"] for g in group]),
            "final_best_fitness": _mean_std([g["final_best_fitness"] for g in group]),
            "final_deceptive": _mean_std([g["final_deceptive"] for g in group]),
            "final_key_rate": _mean_std([g["final_key_rate"] for g in group]),
            "mean_turnover": _mean_std([g["mean_turnover"] for g in group]),
            "late_turnover": _mean_std([g["late_turnover"] for g in group]),
            "final_n_tools": _mean_std([g["final_n_tools"] for g in group]),
        }

    def g(regime: str, cap: int) -> dict:
        return grouped[f"{regime}|k={cap}"]

    low, high = cfg.low_capacity, cfg.high_capacity
    h1_harness = g("harness_first", low)["final_elite_fitness"]["mean"]
    h1_model = g("model_first", low)["final_elite_fitness"]["mean"]
    h2_model = g("model_first", high)["final_elite_fitness"]["mean"]
    h2_harness = g("harness_first", high)["final_elite_fitness"]["mean"]
    # Primary H3 test is high capacity: a one-slot brain cannot reserve the
    # key for rare contexts, so even blind selection drops it. At K=high the
    # frozen controller *can* keep a specialist mapping, so dropping the key
    # is a real greedy error.
    h3_blind = g("harness_first", high)["final_deceptive"]["mean"]
    h3_brain = g("brain_directed", high)["final_deceptive"]["mean"]
    h3_blind_key = g("harness_first", high)["final_key_rate"]["mean"]
    h3_brain_key = g("brain_directed", high)["final_key_rate"]["mean"]
    h3_blind_low = g("harness_first", low)["final_deceptive"]["mean"]
    h3_brain_low = g("brain_directed", low)["final_deceptive"]["mean"]
    h4_coevo = g("coevolution", high)["final_elite_fitness"]["mean"]
    h4_cons = g("coevolution_conservative", high)["final_elite_fitness"]["mean"]
    h4_model = g("model_first", high)["final_elite_fitness"]["mean"]
    h4_harness = g("harness_first", high)["final_elite_fitness"]["mean"]
    h4_turn_coevo = g("coevolution", high)["mean_turnover"]["mean"]
    h4_turn_cons = g("coevolution_conservative", high)["mean_turnover"]["mean"]
    h4_coevo_low = g("coevolution", low)["final_elite_fitness"]["mean"]
    h4_turn_coevo_low = g("coevolution", low)["mean_turnover"]["mean"]
    h4_turn_cons_low = g("coevolution_conservative", low)["mean_turnover"]["mean"]

    hypotheses = {
        "H1": {
            "statement": (
                "Under a tight controller budget, evolving the harness beats "
                "evolving the controller."
            ),
            "held": bool(h1_harness > h1_model),
            "capacity": low,
            "harness_first_elite_fitness": h1_harness,
            "model_first_elite_fitness": h1_model,
            "delta": h1_harness - h1_model,
        },
        "H2": {
            "statement": (
                "Past a capacity threshold, growing the controller beats "
                "growing the harness."
            ),
            "held": bool(h2_model > h2_harness),
            "capacity": high,
            "model_first_elite_fitness": h2_model,
            "harness_first_elite_fitness": h2_harness,
            "delta": h2_model - h2_harness,
        },
        "H3": {
            "statement": (
                "Blind selection on tools beats a myopic brain that edits the "
                "tool it thinks is worst, on a deceptive task."
            ),
            "held": bool(h3_blind > h3_brain),
            "capacity_reported": high,
            "blind_deceptive": h3_blind,
            "brain_directed_deceptive": h3_brain,
            "delta_deceptive": h3_blind - h3_brain,
            "blind_key_retention": h3_blind_key,
            "brain_directed_key_retention": h3_brain_key,
            "low_capacity": {
                "blind_deceptive": h3_blind_low,
                "brain_directed_deceptive": h3_brain_low,
                "held": bool(h3_blind_low > h3_brain_low),
                "note": (
                    "A K=1 controller cannot reserve the key for rare steps, "
                    "so blind selection often drops it too."
                ),
            },
        },
        "H4": {
            "statement": (
                "Coevolution reaches higher fitness but can churn tools "
                "(high turnover) unless selection is conservative."
            ),
            "higher_fitness_than_singles": bool(
                h4_coevo > max(h4_model, h4_harness) or h4_coevo_low > max(
                    g("model_first", low)["final_elite_fitness"]["mean"],
                    g("harness_first", low)["final_elite_fitness"]["mean"],
                )
            ),
            "churn_without_conservative": bool(h4_turn_coevo > h4_turn_cons),
            "held": False,  # filled below
            "coevolution_fitness_high_k": h4_coevo,
            "conservative_fitness_high_k": h4_cons,
            "model_first_fitness_high_k": h4_model,
            "harness_first_fitness_high_k": h4_harness,
            "coevolution_fitness_low_k": h4_coevo_low,
            "coevolution_turnover_high_k": h4_turn_coevo,
            "conservative_turnover_high_k": h4_turn_cons,
            "coevolution_turnover_low_k": h4_turn_coevo_low,
            "conservative_turnover_low_k": h4_turn_cons_low,
        },
    }
    h4 = hypotheses["H4"]
    better_than_one = h4_coevo >= max(h4_model, h4_harness) - 1e-9 or (
        h4_coevo_low
        >= max(
            g("model_first", low)["final_elite_fitness"]["mean"],
            g("harness_first", low)["final_elite_fitness"]["mean"],
        )
        - 1e-9
    )
    h4["held"] = bool(better_than_one and h4["churn_without_conservative"])
    # Keep the two clauses separate so the write-up can discard half of H4.
    h4["fitness_clause"] = bool(better_than_one)
    h4["churn_clause"] = bool(h4["churn_without_conservative"])

    return {
        "grouped": grouped,
        "hypotheses": hypotheses,
    }


def run_experiment(cfg: RunConfig, progress: bool = True) -> dict:
    t0 = time.time()
    cells: list[dict] = []
    jobs: list[tuple[str, int, int]] = []
    for cap in (cfg.low_capacity, cfg.high_capacity):
        for regime in REGIME_NAMES:
            for seed in cfg.seeds:
                jobs.append((regime, cap, seed))
    # Capacity sweep extras (skip cells already in the main grid).
    main_caps = {cfg.low_capacity, cfg.high_capacity}
    for cap in cfg.sweep_capacities:
        if cap in main_caps:
            continue
        for regime in ("model_first", "harness_first"):
            for seed in cfg.seeds:
                jobs.append((regime, cap, seed))

    total = len(jobs)
    for i, (regime, cap, seed) in enumerate(jobs, start=1):
        if progress:
            print(
                f"[{i}/{total}] {regime}  K={cap}  seed={seed}",
                flush=True,
            )
        cells.append(run_cell(cfg, regime, cap, seed))

    agg = aggregate(cells, cfg)
    payload = {
        "meta": {
            "seeds": list(cfg.seeds),
            "n_generations": cfg.n_generations,
            "pop_size": cfg.pop_size,
            "n_parents": cfg.n_parents,
            "low_capacity": cfg.low_capacity,
            "high_capacity": cfg.high_capacity,
            "sweep_capacities": list(cfg.sweep_capacities),
            "eval_steps_per_individual": cfg.eval_steps_per_individual,
            "seed_budget_note": (
                "Every (regime, capacity, seed) cell uses the same generation "
                "count, population size, and evaluation batch size."
            ),
            "seconds": time.time() - t0,
        },
        "hypotheses": agg["hypotheses"],
        "grouped": agg["grouped"],
        "cells": cells,
    }
    return payload


def save_summary(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky per-generation series from a slim companion? Keep full file.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
