"""Publication-style figures for the recorded experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from sim.config import REGIME_COLORS, REGIME_LABELS, RunConfig


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 140,
        }
    )


def _group(payload: dict, regime: str, cap: int) -> dict:
    return payload["grouped"][f"{regime}|k={cap}"]


def plot_fitness_vs_generation(payload: dict, cfg: RunConfig, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    gens = np.arange(cfg.n_generations)
    for ax, cap, title in (
        (axes[0], cfg.low_capacity, f"Low capacity (K={cfg.low_capacity})"),
        (axes[1], cfg.high_capacity, f"High capacity (K={cfg.high_capacity})"),
    ):
        for regime, color in REGIME_COLORS.items():
            g = _group(payload, regime, cap)
            m = np.array(g["elite_fitness_mean"])
            s = np.array(g["elite_fitness_std"])
            ax.plot(gens, m, color=color, lw=2.0, label=REGIME_LABELS[regime])
            ax.fill_between(gens, m - s, m + s, color=color, alpha=0.16, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xlim(0, cfg.n_generations - 1)
    axes[0].set_ylabel("Elite fitness (mean ± 1 sd over seeds)")
    axes[1].legend(loc="lower right", framealpha=0.92)
    fig.suptitle(
        "Same tasks, same seed budget: which subsystem is allowed to change?",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_sweep(payload: dict, cfg: RunConfig, out: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    caps = list(cfg.sweep_capacities)
    for regime, marker in (("model_first", "o"), ("harness_first", "s")):
        means, stds = [], []
        for k in caps:
            g = _group(payload, regime, k)
            means.append(g["final_elite_fitness"]["mean"])
            stds.append(g["final_elite_fitness"]["std"])
        ax.errorbar(
            caps,
            means,
            yerr=stds,
            marker=marker,
            color=REGIME_COLORS[regime],
            lw=2,
            capsize=4,
            label=REGIME_LABELS[regime],
        )
    ax.set_xticks(caps)
    ax.set_xlabel("Controller capacity K (number of prototype slots)")
    ax.set_ylabel("Final elite fitness")
    ax.set_title("Capacity sweep: when does the brain overtake the harness?")
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_turnover(payload: dict, cfg: RunConfig, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    gens = np.arange(cfg.n_generations)
    focus = (
        "harness_first",
        "coevolution",
        "coevolution_conservative",
        "brain_directed",
    )
    for ax, cap, title in (
        (axes[0], cfg.low_capacity, f"Low capacity (K={cfg.low_capacity})"),
        (axes[1], cfg.high_capacity, f"High capacity (K={cfg.high_capacity})"),
    ):
        for regime in focus:
            g = _group(payload, regime, cap)
            m = np.array(g["turnover_mean"])
            s = np.array(g["turnover_std"])
            ax.plot(gens, m, color=REGIME_COLORS[regime], lw=2, label=REGIME_LABELS[regime])
            ax.fill_between(gens, m - s, m + s, color=REGIME_COLORS[regime], alpha=0.14)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_xlim(0, cfg.n_generations - 1)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Elite tool-set Jaccard distance")
    axes[1].legend(loc="upper right", framealpha=0.92)
    fig.suptitle("Tool turnover: open coevolution vs conservative selection", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_harness_vs_brain(payload: dict, cfg: RunConfig, out: Path) -> None:
    """Schematic of the two edit paths plus the measured outcome bars."""
    _style()
    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.15], hspace=0.38, wspace=0.28)
    ax_schematic = fig.add_subplot(gs[0, :])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_schematic.set_xlim(0, 12)
    ax_schematic.set_ylim(0, 4.2)
    ax_schematic.axis("off")
    ax_schematic.set_title(
        "Two ways to change a limb: blind selection vs a brain that thinks it knows",
        pad=8,
    )

    def box(ax, x, y, w, h, text, fc, ec="#222"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            lw=1.2,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#333",
                lw=1.2,
            )
        )

    # Blind path
    box(ax_schematic, 0.3, 2.4, 2.3, 1.1, "Random tool / sensor\nmutation (limbs)", "#F4E3D7")
    box(ax_schematic, 3.4, 2.4, 2.3, 1.1, "Task fitness\n(success − cost)", "#E8F0E8")
    box(ax_schematic, 6.5, 2.4, 2.3, 1.1, "Keep winners\n(blind selection)", "#DCEEEA")
    arrow(ax_schematic, 2.65, 2.95, 3.35, 2.95)
    arrow(ax_schematic, 5.75, 2.95, 6.45, 2.95)
    ax_schematic.text(4.55, 3.7, "Harness-first / coevolution", ha="center", fontsize=9, color="#333")

    # Brain path
    box(ax_schematic, 0.3, 0.35, 2.3, 1.1, "Myopic value of\neach installed tool", "#F7E0EC")
    box(ax_schematic, 3.4, 0.35, 2.3, 1.1, "Edit the tool the\nbrain ranks worst", "#F4D6E6")
    box(ax_schematic, 6.5, 0.35, 2.3, 1.1, "Can drop the delayed\n‘key’ limb", "#E8D0DC")
    arrow(ax_schematic, 2.65, 0.9, 3.35, 0.9)
    arrow(ax_schematic, 5.75, 0.9, 6.45, 0.9)
    ax_schematic.text(4.55, 1.65, "Brain-directed harness edits", ha="center", fontsize=9, color="#333")

    box(ax_schematic, 9.3, 1.2, 2.4, 1.6, "Controller (brain)\ncapacity-capped\npolicy table", "#D6E4F0")

    hy = payload["hypotheses"]
    labels = ["H1  (K=1)", "H2  (K=8)"]
    harness_vals = [
        hy["H1"]["harness_first_elite_fitness"],
        hy["H2"]["harness_first_elite_fitness"],
    ]
    model_vals = [
        hy["H1"]["model_first_elite_fitness"],
        hy["H2"]["model_first_elite_fitness"],
    ]
    x = np.arange(len(labels))
    w = 0.36
    ax_h.bar(
        x - w / 2,
        harness_vals,
        w,
        color=REGIME_COLORS["harness_first"],
        label=REGIME_LABELS["harness_first"],
    )
    ax_h.bar(
        x + w / 2,
        model_vals,
        w,
        color=REGIME_COLORS["model_first"],
        label=REGIME_LABELS["model_first"],
    )
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(labels)
    ax_h.set_ylabel("Final elite fitness")
    ax_h.set_title("H1 / H2: which side should grow first?")
    ax_h.legend(loc="best", fontsize=8)

    d_labels = ["Deceptive score (K=8)", "Elite still has key (K=8)"]
    blind = [hy["H3"]["blind_deceptive"], hy["H3"]["blind_key_retention"]]
    other = [
        hy["H3"]["brain_directed_deceptive"],
        hy["H3"]["brain_directed_key_retention"],
    ]
    x = np.arange(len(d_labels))
    ax_d.bar(
        x - w / 2,
        blind,
        w,
        color=REGIME_COLORS["harness_first"],
        label="Blind harness selection",
    )
    ax_d.bar(
        x + w / 2,
        other,
        w,
        color=REGIME_COLORS["brain_directed"],
        label="Myopic brain-directed edits",
    )
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(d_labels)
    ax_d.set_title("H3: delayed-reward tool vs greedy editor")
    ax_d.legend(loc="best", fontsize=8)

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_key_retention(payload: dict, cfg: RunConfig, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    gens = np.arange(cfg.n_generations)
    for ax, cap, title in (
        (axes[0], cfg.low_capacity, f"Low capacity (K={cfg.low_capacity})"),
        (axes[1], cfg.high_capacity, f"High capacity (K={cfg.high_capacity})"),
    ):
        for regime in ("harness_first", "brain_directed", "coevolution"):
            g = _group(payload, regime, cap)
            m = np.array(g["key_rate_mean"])
            s = np.array(g["key_rate_std"])
            ax.plot(gens, m, color=REGIME_COLORS[regime], lw=2, label=REGIME_LABELS[regime])
            ax.fill_between(gens, m - s, m + s, color=REGIME_COLORS[regime], alpha=0.14)
        ax.set_title(title)
        ax.set_xlabel("Generation")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, cfg.n_generations - 1)
    axes[0].set_ylabel("Elite fraction still carrying the ‘key’ tool")
    axes[1].legend(loc="best", framealpha=0.92)
    fig.suptitle("Deceptive landscape: does a greedy brain drop the delayed-reward limb?", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_all_plots(payload: dict, cfg: RunConfig, results_dir: Path) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        results_dir / "fitness_vs_generation.png",
        results_dir / "capacity_sweep.png",
        results_dir / "tool_turnover.png",
        results_dir / "harness_vs_brain.png",
        results_dir / "key_retention.png",
    ]
    plot_fitness_vs_generation(payload, cfg, paths[0])
    plot_capacity_sweep(payload, cfg, paths[1])
    plot_turnover(payload, cfg, paths[2])
    plot_harness_vs_brain(payload, cfg, paths[3])
    plot_key_retention(payload, cfg, paths[4])
    return paths
