"""Entry point: ``python -m sim.run`` from a venv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sim.config import DEFAULT, RunConfig, smoke_config
from sim.experiment import run_experiment, save_summary
from sim.plot import write_all_plots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evolve controllers and/or harnesses on synthetic agent tasks."
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run (2 seeds, 6 generations) to check the checkout.",
    )
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--gens", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="Results directory.")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    cfg = smoke_config() if args.smoke else DEFAULT
    updates = {}
    if args.seeds is not None:
        updates["seeds"] = tuple(args.seeds)
    if args.gens is not None:
        updates["n_generations"] = args.gens
    if args.pop is not None:
        updates["pop_size"] = args.pop
    if args.out is not None:
        updates["results_dir"] = args.out
    if updates:
        # dataclass replace
        from dataclasses import replace

        cfg = replace(cfg, **updates)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    print(
        f"Running {len(cfg.seeds)} seeds × {cfg.n_generations} gens × pop {cfg.pop_size}",
        flush=True,
    )
    payload = run_experiment(cfg, progress=True)
    out = Path(cfg.results_dir)
    save_summary(payload, out / "summary.json")
    paths = write_all_plots(payload, cfg, out)
    hy = payload["hypotheses"]
    print("\n=== Hypotheses (label held only after this run) ===")
    for key in ("H1", "H2", "H3", "H4"):
        item = hy[key]
        print(f"{key} held={item['held']}: {item['statement']}")
    print(f"\nWrote {out / 'summary.json'}")
    for p in paths:
        print(f"Wrote {p}")
    print(f"Elapsed {payload['meta']['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
