"""Frozen tool / sensor catalog. Tools are limbs: cost, capability, noise."""

from __future__ import annotations

import numpy as np

from sim.config import RunConfig


# Catalog order is part of the genome. Do not reorder without bumping results.
TOOL_NAMES = (
    "bare",
    "generalist",
    "spec0",
    "spec1",
    "spec2",
    "spec3",
    "spec4",
    "spec5",
    "bait",
    "key",
    "lure",
    "heavy",
    "noisy_spec",
    "junk",
)

BARE_ID = 0
GENERALIST_ID = 1
BAIT_ID = 8
KEY_ID = 9
LURE_ID = 10
HEAVY_ID = 11
JUNK_ID = 13
SPECIALIST_IDS = (2, 3, 4, 5, 6, 7)


def tool_costs() -> np.ndarray:
    # Maintenance cost paid every generation if the tool is installed.
    return np.array(
        [
            0.010,  # bare
            0.12,  # generalist
            0.055,
            0.055,
            0.055,
            0.055,
            0.055,
            0.055,  # specialists
            0.045,  # bait: cheap, looks good on common niches
            0.085,  # key: needed on rare / delayed niches
            0.04,  # lure
            0.22,  # heavy
            0.08,  # noisy_spec
            0.12,  # junk
        ],
        dtype=np.float64,
    )


def tool_base_noise() -> np.ndarray:
    return np.array(
        [
            0.08,
            0.10,
            0.07,
            0.07,
            0.07,
            0.07,
            0.07,
            0.07,
            0.06,
            0.18,  # key is noisy on the trials a myopic brain sees
            0.08,
            0.12,
            0.28,
            0.15,
        ],
        dtype=np.float64,
    )


def tool_capabilities(n_niches: int = 6) -> np.ndarray:
    """capability[tool, niche] in [0, 1] before noise."""
    cap = np.zeros((len(TOOL_NAMES), n_niches), dtype=np.float64)
    cap[BARE_ID] = 0.18
    cap[GENERALIST_ID] = 0.40
    for i, tid in enumerate(SPECIALIST_IDS):
        cap[tid] = 0.10
        cap[tid, i] = 0.96
    # Bait: excellent on common niches 0-2, nearly useless on rare 3-5.
    cap[BAIT_ID] = np.array([0.90, 0.88, 0.86, 0.06, 0.06, 0.05])
    # Key: weak on common (looks bad), strong on rare jackpot niches.
    cap[KEY_ID] = np.array([0.16, 0.16, 0.15, 0.94, 0.93, 0.92])
    cap[LURE_ID] = np.array([0.72, 0.70, 0.18, 0.10, 0.10, 0.08])
    cap[HEAVY_ID] = 0.68
    cap[NOISY := 12] = 0.14
    cap[NOISY, 0] = 0.84
    cap[JUNK_ID] = 0.30
    return cap


def niche_features(n_features: int = 6, n_niches: int = 6) -> np.ndarray:
    """Each niche is a nearly one-hot feature pattern plus a shared bit."""
    feats = np.zeros((n_niches, n_features), dtype=np.float64)
    for i in range(n_niches):
        feats[i, i] = 1.0
        feats[i, i % n_features] = 1.0
    # Shared climate bit: common niches share feature 0 faintly already.
    return feats


def sensor_cost() -> float:
    return 0.03


def build_catalog(cfg: RunConfig) -> dict:
    caps = tool_capabilities(cfg.n_niches)
    costs = tool_costs()
    noise = tool_base_noise()
    assert caps.shape[0] == cfg.n_catalog == len(TOOL_NAMES)
    return {
        "names": TOOL_NAMES,
        "costs": costs,
        "base_noise": noise,
        "capability": caps,
        "niche_features": niche_features(cfg.n_features, cfg.n_niches),
        "sensor_cost": sensor_cost(),
    }
