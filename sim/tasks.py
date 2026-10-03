"""Synthetic tasks: foraging, contextual bandit, deceptive delayed reward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.agent import Genome, harness_cost, spam_penalty
from sim.catalog import KEY_ID, build_catalog
from sim.config import RunConfig


@dataclass
class Batch:
    features: np.ndarray  # (N, F)
    niche: np.ndarray  # (N,) int, -1 = empty patch
    scale: np.ndarray  # (N,)
    myopic: np.ndarray  # (N,) bool — visible to a greedy brain


def _sample_niches(
    rng: np.random.Generator, n: int, n_niches: int, common: bool
) -> np.ndarray:
    if common:
        return rng.integers(0, 3, size=n)
    return rng.integers(3, n_niches, size=n)


def make_forage_batch(rng: np.random.Generator, cfg: RunConfig, catalog: dict) -> Batch:
    n = cfg.forage_episodes * cfg.forage_horizon
    empty = rng.random(n) < 0.18
    # Mostly common resources, some rare.
    rare = rng.random(n) < 0.18
    niche = np.where(
        rare,
        rng.integers(3, cfg.n_niches, size=n),
        rng.integers(0, 3, size=n),
    )
    niche = niche.astype(np.int64)
    niche[empty] = -1
    feats = np.zeros((n, cfg.n_features), dtype=np.float64)
    valid = niche >= 0
    feats[valid] = catalog["niche_features"][niche[valid]]
    # Empty patches still have weak clutter features.
    if np.any(~valid):
        feats[~valid] = rng.random((int(np.sum(~valid)), cfg.n_features)) < 0.15
    scale = np.ones(n, dtype=np.float64)
    scale[niche >= 3] = 1.6
    myopic = niche < 3  # empty and common look "ordinary"
    return Batch(features=feats, niche=niche, scale=scale, myopic=myopic)


def make_bandit_batch(rng: np.random.Generator, cfg: RunConfig, catalog: dict) -> Batch:
    n = cfg.bandit_episodes
    niche = rng.integers(0, cfg.n_niches, size=n)
    feats = catalog["niche_features"][niche].copy()
    # Small feature noise in the world itself.
    flip = rng.random(feats.shape) < 0.04
    feats = np.where(flip, 1.0 - feats, feats)
    scale = np.ones(n, dtype=np.float64)
    myopic = niche < 3
    return Batch(features=feats, niche=niche, scale=scale, myopic=myopic)


def make_deceptive_batch(
    rng: np.random.Generator, cfg: RunConfig, catalog: dict
) -> Batch:
    h = cfg.deceptive_common + cfg.deceptive_rare
    n = cfg.deceptive_episodes * h
    niche = np.empty(n, dtype=np.int64)
    myopic = np.zeros(n, dtype=bool)
    for ep in range(cfg.deceptive_episodes):
        sl = slice(ep * h, (ep + 1) * h)
        niche[sl] = np.concatenate(
            [
                _sample_niches(rng, cfg.deceptive_common, cfg.n_niches, True),
                _sample_niches(rng, cfg.deceptive_rare, cfg.n_niches, False),
            ]
        )
        myopic[sl] = np.concatenate(
            [
                np.ones(cfg.deceptive_common, dtype=bool),
                np.zeros(cfg.deceptive_rare, dtype=bool),
            ]
        )
    feats = catalog["niche_features"][niche].copy()
    scale = np.ones(n, dtype=np.float64)
    scale[~myopic] = cfg.rare_reward_scale
    return Batch(features=feats, niche=niche, scale=scale, myopic=myopic)


def make_generation_batches(
    rng: np.random.Generator, cfg: RunConfig
) -> dict[str, Batch]:
    catalog = build_catalog(cfg)
    return {
        "forage": make_forage_batch(rng, cfg, catalog),
        "bandit": make_bandit_batch(rng, cfg, catalog),
        "deceptive": make_deceptive_batch(rng, cfg, catalog),
    }


def choose_tools(g: Genome, batch: Batch, cfg: RunConfig) -> np.ndarray:
    """Vectorized nearest-prototype controller. Missing sensors read as 0.5."""
    n, f = batch.features.shape
    present = g.sensor_on.astype(np.float64)
    noise = g.sensor_noise
    # Expected observation after bit-flip noise; absent sensors are 0.5.
    signal = batch.features * (1.0 - 2.0 * noise) + noise
    obs = signal * present + 0.5 * (1.0 - present)
    # Distance only on sensed coordinates; unsensed dims do not discriminate.
    weight = present[None, None, :]  # (1, 1, F)
    proto = g.prototypes[None, :, :]  # (1, K, F)
    diff = obs[:, None, :] - proto
    dist = np.sum(weight * diff * diff, axis=-1)  # (N, K)
    # If no sensors, every slot looks the same — fall back to slot 0 (dumb default).
    if float(present.sum()) < 1e-9:
        winner = np.zeros(n, dtype=np.int64)
    else:
        winner = np.argmin(dist, axis=1)
    wanted = g.slot_tool[winner]
    on = g.tool_on
    chosen = wanted.copy()
    missing = ~on[wanted]
    if np.any(missing):
        # Fall back to a present tool; prefer bare hands if installed.
        fallback = int(np.flatnonzero(on)[0]) if on.any() else 0
        if on[0]:
            fallback = 0
        chosen[missing] = fallback
    return chosen.astype(np.int64)


def expected_success(
    tools: np.ndarray, batch: Batch, g: Genome, catalog: dict
) -> np.ndarray:
    cap = catalog["capability"]
    p = np.zeros(tools.shape[0], dtype=np.float64)
    valid = batch.niche >= 0
    if np.any(valid):
        p[valid] = cap[tools[valid], batch.niche[valid]] * (1.0 - g.tool_noise[tools[valid]])
    # Empty patch: bare hands are fine; other tools waste the step.
    if np.any(~valid):
        empty_tools = tools[~valid]
        p[~valid] = np.where(empty_tools == 0, 0.42, 0.10)
    return np.clip(p, 0.0, 1.0)


def evaluate_batch(
    g: Genome, batch: Batch, cfg: RunConfig, catalog: dict
) -> dict:
    tools = choose_tools(g, batch, cfg)
    p = expected_success(tools, batch, g, catalog)
    step = p * batch.scale - cfg.use_fee
    apparent = np.full(cfg.n_catalog, -0.85, dtype=np.float64)
    for tid in np.flatnonzero(g.tool_on):
        used = (tools == tid) & batch.myopic
        if np.any(used):
            apparent[tid] = float(step[used].mean())
        elif g.tool_on[tid]:
            # Unused on the myopic slice: the greedy brain treats it as dead weight.
            apparent[tid] = -0.55 - 0.4 * float(catalog["costs"][tid])
    myopic_step = step[batch.myopic] if np.any(batch.myopic) else step
    return {
        "mean_step": float(step.mean()),
        "myopic_step": float(myopic_step.mean()),
        "success": float(p.mean()),
        "tools": tools,
        "apparent": apparent,
        "key_used": float(np.mean(tools == KEY_ID)),
    }


def evaluate_genome(g: Genome, batches: dict[str, Batch], cfg: RunConfig) -> dict:
    catalog = build_catalog(cfg)
    parts = {name: evaluate_batch(g, batch, cfg, catalog) for name, batch in batches.items()}
    success = (
        parts["forage"]["mean_step"]
        + parts["bandit"]["mean_step"]
        + parts["deceptive"]["mean_step"]
    ) / 3.0
    myopic_success = (
        parts["forage"]["myopic_step"]
        + parts["bandit"]["myopic_step"]
        + parts["deceptive"]["myopic_step"]
    ) / 3.0
    cost = harness_cost(g, cfg)
    spam = spam_penalty(g, cfg)
    penalty = cfg.cost_weight * cost + cfg.spam_weight * spam
    fitness = success - penalty
    myopic_fitness = myopic_success - penalty
    apparent = (
        parts["forage"]["apparent"]
        + parts["bandit"]["apparent"]
        + parts["deceptive"]["apparent"]
    ) / 3.0
    return {
        "fitness": float(fitness),
        "myopic_fitness": float(myopic_fitness),
        "success": float(success),
        "myopic_success": float(myopic_success),
        "forage": parts["forage"]["mean_step"],
        "bandit": parts["bandit"]["mean_step"],
        "deceptive": parts["deceptive"]["mean_step"],
        "cost": float(cost),
        "spam": float(spam),
        "n_tools": int(g.tool_on.sum()),
        "n_sensors": int(g.sensor_on.sum()),
        "has_key": bool(g.tool_on[KEY_ID]),
        "has_bait": bool(g.tool_on[8]),
        "key_used": parts["deceptive"]["key_used"],
        "apparent": apparent,
        "tool_set": g.tool_set(),
    }
