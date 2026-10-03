# Experiment

This is a synthetic evolutionary simulation. It is not a user study and it does not call a language model. Numbers below are from the recorded run that wrote `results/summary.json` (8 seeds, 48 generations, population 32). Re-run with `python -m sim.run` from a venv to reproduce.

## Question

An agent is a **controller** (brain) inside a **harness** of tools and sensors (limbs). Tools have a maintenance cost, a capability profile across contexts, and a failure rate. Fitness is expected task success minus harness cost minus a spam penalty, so installing every tool is not free.

Should the brain decide which tools to keep, or should the harness change under blind variation and selection — the way limbs did, before nervous systems could redesign them?

## Hypotheses (stated before looking at this run)

- **H1.** Under a tight controller budget, evolving the harness beats evolving the controller.
- **H2.** Past a capacity threshold, growing the controller beats growing the harness.
- **H3.** Blind selection on tools beats a myopic brain that edits the tool it thinks is worst, on a deceptive task.
- **H4.** Coevolution reaches higher fitness but can churn tools (high turnover) unless selection is conservative.

## Setup

### World

Six resource **niches**. Niches 0–2 are common; niches 3–5 are rare. Each niche has a nearly one-hot feature vector. A catalog of 14 tools (bare hands, a cheap generalist, six specialists, a “bait” tool that looks good on common niches, a “key” tool that pays on rare niches, and several traps/junk) plus one sensor per feature.

Each tool is a limb: **cost**, **capability[niche]**, **noise**. Sensors that are off return 0.5 (no information). Sensors that are on return a noisy expected observation.

### Controller

A capacity-`K` prototype table. Each slot stores a feature prototype and a single preferred tool id. The agent picks the nearest prototype on the sensed coordinates and uses that slot’s tool if the tool is installed; otherwise it falls back to bare hands. `K=1` is a primitive creature (one default action). `K=8` can in principle store one mapping per niche.

### Tasks (same for every regime)

1. **Foraging with partial sensing.** 16 episodes × 8 patches. Some patches are empty (using a fancy tool wastes the step). Mix of common and rare resources.
2. **Contextual tool-choice bandit.** 24 single-step contexts, niches drawn uniformly. The right specialist depends on the context.
3. **Deceptive / delayed reward.** 12 episodes × (10 common steps + 2 rare jackpot steps). Rare steps have reward scale 5. A myopic value estimate sees only the common slice.

Fitness of an individual is the mean of the three task scores, minus `0.32 × harness_cost` minus a spam term for toolkits larger than eight tools.

### Regimes

Every cell uses the same seed budget: 48 generations, population 32, 8 parents, 296 evaluation steps per individual per generation. Shared evaluation batches are drawn from `(seed, generation)` so individuals in a generation see the same worlds.

| Regime | What mutates | What is frozen | Selection objective |
| --- | --- | --- | --- |
| `model_first` | Controller | A complete field kit (bare + 6 specialists + key + all sensors) | True fitness |
| `harness_first` | Tools and sensors | One random capacity-`K` controller | True fitness |
| `coevolution` | Both | Nothing | True fitness |
| `brain_directed` | Harness, but the myopic value estimate chooses which installed tool to edit | Same frozen controller as harness-first | **Myopic** fitness (common / early steps only) |
| `coevolution_conservative` | Both, lower harness mutation, 6 elites | Nothing | True fitness |

Brain-directed is the “is the brain smart enough to know what to evolve” arm. It is allowed to drop the tool it ranks worst on the common slice, and it is selected on that same myopic score. True (full-horizon) fitness is still recorded for comparison.

### What was actually run

```
python -m sim.run
```

- Seeds: `0, 1, 2, 3, 4, 5, 6, 7`
- Generations: 48
- Population: 32
- Capacities in the main grid: `K=1` and `K=8` × all five regimes
- Capacity sweep extras: `K=2, 4` × `{model_first, harness_first}`
- Wall time on this machine: 49.2 s
- Raw series: `results/summary.json`

No network calls. Numpy `Generator` streams are seeded with `SeedSequence([seed, regime_id, K, 0xA11CE])`.

## Measured outcome

Elite fitness is the mean true fitness of the top 8 individuals, then averaged across the 8 seeds. Values are mean ± 1 sample standard deviation.

### H1 — held

At `K=1`, harness-first elite fitness **0.616 ± 0.051** beat model-first **0.148 ± 0.025** (delta **+0.468**).

A one-slot brain cannot represent a context→tool map. Evolving the controller on a frozen complete kit goes nowhere. Evolving the kit around that dumb default finds a cheap one-tool body (elite toolkit size 1.00). That is morphological computation in miniature: the limb does the work the brain cannot.

### H2 — held

At `K=8`, model-first **0.697 ± 0.017** beat harness-first **0.591 ± 0.105** (delta **+0.106**).

The capacity sweep is the cleaner picture:

| K | Model-first | Harness-first |
| --- | --- | --- |
| 1 | 0.148 ± 0.025 | 0.616 ± 0.051 |
| 2 | 0.320 ± 0.025 | 0.522 ± 0.140 |
| 4 | 0.596 ± 0.083 | 0.573 ± 0.104 |
| 8 | 0.697 ± 0.017 | 0.591 ± 0.105 |

The curves cross between `K=2` and `K=4`. Past that threshold, growing the controller on a fixed complete harness overtakes growing the harness around a frozen brain. See `results/capacity_sweep.png`.

### H3 — held

Primary comparison is `K=8`, where a controller *can* reserve a tool for rare contexts. Blind harness-first deceptive score **0.981 ± 0.182** beat brain-directed **0.677 ± 0.189** (delta **+0.304**). Elite key-tool retention: **0.375** (blind) vs **0.000** (brain-directed).

The myopic editor consistently treats the key as dead weight (it looks bad on common steps) and drops it. Blind mutation plus *true* fitness keeps it in a minority of elites and, more importantly, does not systematically hunt it.

At `K=1` the deceptive gap also favored blind selection (0.941 vs 0.581), even though both arms dropped the key. A one-slot brain cannot save the key for the jackpot, so H3 at `K=1` is a weaker test. The `K=8` comparison is the one H3 was written for.

### H4 — held on churn; fitness clause is capacity-dependent

| Cell | Elite fitness | Mean tool-set turnover |
| --- | --- | --- |
| Coevolution, K=1 | 0.664 ± 0.014 | 0.099 ± 0.016 |
| Conservative coevolution, K=1 | 0.556 ± 0.102 | 0.067 ± 0.009 |
| Coevolution, K=8 | 0.685 ± 0.035 | 0.122 ± 0.024 |
| Conservative coevolution, K=8 | 0.660 ± 0.058 | 0.079 ± 0.017 |
| Model-first, K=8 | 0.697 ± 0.017 | 0.000 (harness frozen) |
| Harness-first, K=8 | 0.591 ± 0.105 | 0.091 ± 0.024 |

- **Fitness vs evolving only one side.** At `K=1`, coevolution (0.664) beat both singles (harness 0.616, model 0.148). At `K=8`, coevolution (0.685) beat harness-first (0.591) but did **not** beat model-first (0.697). The two high-K means overlap within a standard deviation. Do not read this as “coevolution always wins.”
- **Churn.** Open coevolution turned tools over faster than conservative selection at both capacities (0.122 vs 0.079 at `K=8`; 0.099 vs 0.067 at `K=1`). Conservative selection paid a bit of fitness at `K=1` and was roughly tied at `K=8`.

So H4’s churn clause held. The “higher fitness” clause held against single-sided evolution at low capacity, and only against harness-first at high capacity.

Open coevolution often collapsed to a one- or two-tool body and dropped the key. A costly generalist limb can collect the delayed jackpot without a specialized key — another morphological shortcut. That is why key retention is not the only story in `results/key_retention.png`.

## Plots

- `results/fitness_vs_generation.png` — elite fitness vs generation, both capacities, all five regimes
- `results/capacity_sweep.png` — H1/H2 crossover
- `results/tool_turnover.png` — Jaccard distance of elite tool sets
- `results/key_retention.png` — fraction of elites still carrying the key
- `results/harness_vs_brain.png` — schematic of blind vs brain-directed edits, plus H1–H3 bars

## What this does not show

It does not show that real LLM agents should freeze the weights and evolve tools, or the reverse. The controller is a `K`-slot table, not a transformer. The tasks are three synthetic landscapes. The result is: **in this toy, the harness is the thing to evolve while the brain is primitive; past a capacity threshold the brain takes over; and a greedy brain that edits the limb it dislikes can lock out a delayed-reward tool that blind selection would sometimes keep.**
