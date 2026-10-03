"""Experiment knobs. Defaults match the recorded run in docs/experiment.md."""

from __future__ import annotations

from dataclasses import dataclass, field


REGIME_NAMES = (
    "model_first",
    "harness_first",
    "coevolution",
    "brain_directed",
    "coevolution_conservative",
)

REGIME_LABELS = {
    "model_first": "Model-first (evolve controller)",
    "harness_first": "Harness-first (evolve tools)",
    "coevolution": "Coevolution",
    "brain_directed": "Brain-directed harness edits",
    "coevolution_conservative": "Conservative coevolution",
}

# Colorblind-safe Okabe–Ito palette
REGIME_COLORS = {
    "model_first": "#0072B2",
    "harness_first": "#D55E00",
    "coevolution": "#009E73",
    "brain_directed": "#CC79A7",
    "coevolution_conservative": "#E69F00",
}


@dataclass(frozen=True)
class RunConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    n_generations: int = 48
    pop_size: int = 32
    n_elites: int = 2
    n_parents: int = 8
    low_capacity: int = 1
    high_capacity: int = 8
    sweep_capacities: tuple[int, ...] = (1, 2, 4, 8)
    n_features: int = 6
    n_niches: int = 6
    n_catalog: int = 14
    cost_weight: float = 0.32
    spam_weight: float = 0.10
    use_fee: float = 0.025
    controller_mut: float = 0.55
    harness_mut: float = 0.50
    conservative_harness_mut: float = 0.16
    conservative_elites: int = 6
    results_dir: str = "results"
    # Shared evaluation budget per individual per generation (episode-steps).
    forage_episodes: int = 16
    forage_horizon: int = 8
    bandit_episodes: int = 24
    deceptive_episodes: int = 12
    deceptive_common: int = 10
    deceptive_rare: int = 2
    rare_reward_scale: float = 5.0

    extra: dict = field(default_factory=dict)

    @property
    def eval_steps_per_individual(self) -> int:
        return (
            self.forage_episodes * self.forage_horizon
            + self.bandit_episodes
            + self.deceptive_episodes * (self.deceptive_common + self.deceptive_rare)
        )


DEFAULT = RunConfig()


def smoke_config() -> RunConfig:
    """Tiny run for a checkout sanity check."""
    return RunConfig(
        seeds=(0, 1),
        n_generations=6,
        pop_size=12,
        n_parents=4,
        sweep_capacities=(1, 8),
        forage_episodes=6,
        forage_horizon=6,
        bandit_episodes=8,
        deceptive_episodes=4,
        deceptive_common=5,
        deceptive_rare=2,
    )
