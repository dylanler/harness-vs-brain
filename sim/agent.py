"""Capacity-limited controller + mutable harness genome."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.catalog import BARE_ID, KEY_ID, SPECIALIST_IDS, build_catalog
from sim.config import RunConfig


@dataclass
class Genome:
    prototypes: np.ndarray  # (K, F)
    slot_tool: np.ndarray  # (K,) int — which catalog tool each slot wants
    tool_on: np.ndarray  # (C,) bool
    tool_noise: np.ndarray  # (C,)
    sensor_on: np.ndarray  # (F,) bool
    sensor_noise: np.ndarray  # (F,)

    def copy(self) -> "Genome":
        return Genome(
            prototypes=self.prototypes.copy(),
            slot_tool=self.slot_tool.copy(),
            tool_on=self.tool_on.copy(),
            tool_noise=self.tool_noise.copy(),
            sensor_on=self.sensor_on.copy(),
            sensor_noise=self.sensor_noise.copy(),
        )

    def tool_set(self) -> frozenset[int]:
        return frozenset(np.flatnonzero(self.tool_on).tolist())


def canonical_harness(cfg: RunConfig) -> dict:
    """A complete-but-costly field kit. Frozen in the model-first regime."""
    catalog = build_catalog(cfg)
    c, f = cfg.n_catalog, cfg.n_features
    tool_on = np.zeros(c, dtype=bool)
    tool_on[BARE_ID] = True
    tool_on[list(SPECIALIST_IDS)] = True
    tool_on[KEY_ID] = True
    sensor_on = np.ones(f, dtype=bool)
    return {
        "tool_on": tool_on,
        "tool_noise": catalog["base_noise"].copy(),
        "sensor_on": sensor_on,
        "sensor_noise": np.full(f, 0.08, dtype=np.float64),
    }


def apply_harness(g: Genome, harness: dict) -> None:
    g.tool_on = harness["tool_on"].copy()
    g.tool_noise = harness["tool_noise"].copy()
    g.sensor_on = harness["sensor_on"].copy()
    g.sensor_noise = harness["sensor_noise"].copy()


def apply_controller(g: Genome, brain: Genome) -> None:
    g.prototypes = brain.prototypes.copy()
    g.slot_tool = brain.slot_tool.copy()


def random_genome(rng: np.random.Generator, cfg: RunConfig, capacity: int) -> Genome:
    catalog = build_catalog(cfg)
    c, f = cfg.n_catalog, cfg.n_features
    tool_on = rng.random(c) < 0.34
    tool_on[BARE_ID] = True
    # Keep the initial body from being a kitchen-sink.
    extra = np.flatnonzero(tool_on)
    if extra.size > 6:
        drop = rng.choice(extra[extra != BARE_ID], size=extra.size - 6, replace=False)
        tool_on[drop] = False
    sensor_on = rng.random(f) < 0.40
    return Genome(
        prototypes=rng.normal(0.45, 0.30, size=(capacity, f)),
        slot_tool=rng.integers(0, c, size=capacity),
        tool_on=tool_on,
        tool_noise=np.clip(
            catalog["base_noise"] + rng.normal(0.0, 0.03, size=c), 0.04, 0.55
        ),
        sensor_on=sensor_on,
        sensor_noise=np.clip(rng.normal(0.10, 0.04, size=f), 0.02, 0.40),
    )


def harness_cost(g: Genome, cfg: RunConfig) -> float:
    catalog = build_catalog(cfg)
    tool_cost = float(np.sum(catalog["costs"] * g.tool_on))
    # More reliable tools cost extra — a limb that never fails is expensive.
    reliability_tax = float(np.sum(g.tool_on * (0.22 - g.tool_noise).clip(0.0, 0.20)))
    sensor_cost = float(np.sum(g.sensor_on) * catalog["sensor_cost"])
    return tool_cost + 0.35 * reliability_tax + sensor_cost


def spam_penalty(g: Genome, cfg: RunConfig) -> float:
    n_tools = int(g.tool_on.sum())
    n_sensors = int(g.sensor_on.sum())
    # A full specialist set is allowed; junk beyond that is spam.
    extra_tools = max(0, n_tools - 8)
    extra_sensors = max(0, n_sensors - cfg.n_features)
    return extra_tools * 0.10 + extra_sensors * 0.04


def mutate_controller(g: Genome, rng: np.random.Generator, scale: float = 1.0) -> None:
    k, f = g.prototypes.shape
    c = int(g.tool_on.size)
    g.prototypes += rng.normal(0.0, 0.18 * scale, g.prototypes.shape)
    np.clip(g.prototypes, -0.2, 1.2, out=g.prototypes)
    if rng.random() < 0.55:
        slot = int(rng.integers(0, k))
        g.slot_tool[slot] = int(rng.integers(0, c))
    # Constructive mutation: bind one slot to a single feature / specialist.
    if rng.random() < 0.35:
        slot = int(rng.integers(0, k))
        feat = int(rng.integers(0, f))
        g.prototypes[slot] = 0.15
        g.prototypes[slot, feat] = 1.0
        g.slot_tool[slot] = int(SPECIALIST_IDS[feat]) if feat < len(SPECIALIST_IDS) else int(
            rng.integers(0, c)
        )


def mutate_harness_blind(g: Genome, rng: np.random.Generator, intensity: float) -> None:
    c = g.tool_on.size
    f = g.sensor_on.size
    n_flips = 1 + int(rng.random() < intensity)
    for _ in range(n_flips):
        if rng.random() < 0.72:
            idx = int(rng.integers(0, c))
            if idx == BARE_ID and g.tool_on[idx] and rng.random() < 0.7:
                continue
            g.tool_on[idx] = ~g.tool_on[idx]
        else:
            idx = int(rng.integers(0, f))
            g.sensor_on[idx] = ~g.sensor_on[idx]
    if rng.random() < 0.45:
        idx = int(rng.integers(0, c))
        g.tool_noise[idx] = float(
            np.clip(g.tool_noise[idx] + rng.normal(0.0, 0.06), 0.04, 0.55)
        )
    if not g.tool_on.any():
        g.tool_on[BARE_ID] = True


def mutate_harness_brain(
    g: Genome, rng: np.random.Generator, apparent: np.ndarray, intensity: float
) -> None:
    """Mutate the harness part the myopic value estimate likes least."""
    on = np.flatnonzero(g.tool_on)
    if on.size == 0:
        mutate_harness_blind(g, rng, intensity)
        return
    values = apparent[on].astype(np.float64)
    # Softmin: lowest apparent value is most likely to be edited.
    logits = -values / 0.12
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    target = int(on[rng.choice(on.size, p=p)])
    # Greedy edit: drop or replace the "worst" installed tool.
    if rng.random() < 0.72:
        if target != BARE_ID or on.size == 1:
            g.tool_on[target] = False
        # Sometimes swap in something else the brain has not priced.
        if rng.random() < 0.55:
            off = np.flatnonzero(~g.tool_on)
            if off.size:
                g.tool_on[int(rng.choice(off))] = True
    else:
        g.tool_noise[target] = float(
            np.clip(g.tool_noise[target] + rng.normal(0.0, 0.08), 0.04, 0.55)
        )
    if rng.random() < 0.25:
        idx = int(rng.integers(0, g.sensor_on.size))
        g.sensor_on[idx] = ~g.sensor_on[idx]
    if not g.tool_on.any():
        g.tool_on[BARE_ID] = True


def apply_mutation(
    child: Genome,
    rng: np.random.Generator,
    regime: str,
    cfg: RunConfig,
    apparent: np.ndarray | None,
) -> None:
    if regime == "model_first":
        if rng.random() < cfg.controller_mut:
            mutate_controller(child, rng)
        return
    if regime == "harness_first":
        if rng.random() < cfg.harness_mut:
            mutate_harness_blind(child, rng, cfg.harness_mut)
        return
    if regime == "brain_directed":
        if rng.random() < cfg.harness_mut:
            if apparent is None:
                mutate_harness_blind(child, rng, cfg.harness_mut)
            else:
                mutate_harness_brain(child, rng, apparent, cfg.harness_mut)
        return
    if regime == "coevolution":
        if rng.random() < cfg.controller_mut:
            mutate_controller(child, rng)
        if rng.random() < cfg.harness_mut:
            mutate_harness_blind(child, rng, cfg.harness_mut)
        return
    if regime == "coevolution_conservative":
        if rng.random() < cfg.controller_mut * 0.7:
            mutate_controller(child, rng, scale=0.65)
        if rng.random() < cfg.conservative_harness_mut:
            mutate_harness_blind(child, rng, cfg.conservative_harness_mut)
        return
    raise ValueError(f"unknown regime {regime}")
