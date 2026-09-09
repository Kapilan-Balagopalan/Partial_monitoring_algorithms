"""
Multi-algorithm comparison on the Apple Tasting game (contextual setting).

Algorithms included:
  - Random           — uniform baseline
  - PGTS             — Polya-Gamma Thompson Sampling
  - STAP-Helmbolt    — Helmbold et al.
  - SquareCB.PMSide  — IGW + water-transfer (this work)

Run:
    py run_comparison_at.py

Output:
    comparison_at_results.npy   (dict of {name: (N_SEEDS, HORIZON) arrays})
    comparison_at_regret.png
"""

import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import apple_tasting
from synthetic_data import LinearContexts
from evaluation_contextual import Evaluation_contextual
from squarecb_pmside import SquareCBPMSide
from STAP_Helmbolt import STAP_Helmbolt
from PGTS import PGTS
from random_algo import Random

# ── Experiment settings ────────────────────────────────────────────────────────
HORIZON = 3000
N_SEEDS = 5
D       = 2          # context dimension
GAMMA   = 1.0        # SquareCB.PMSide: gamma_t = GAMMA * sqrt(A t), A = number of arms
LBD     = 0.05       # ridge regularisation (all ridge-regression algorithms)
W       = np.array([0.6, 0.4])   # LinearContexts outcome-probability weights

# ── Game / evaluator ──────────────────────────────────────────────────────────
game        = apple_tasting(restructure_game=False)
context_gen = LinearContexts(W)
evaluator   = Evaluation_contextual(HORIZON)

# ── Algorithm registry ────────────────────────────────────────────────────────
# Each entry: (display_name, factory_fn)
# factory_fn() returns a fresh algorithm instance.
ALGOS = [
    ("Random",
     lambda: Random(game, HORIZON)),

    ("PGTS",
     lambda: PGTS(game, D)),

    ("STAP-Helmbolt",
     lambda: STAP_Helmbolt(game, D)),

    ("SquareCB.PMSide",
     lambda: SquareCBPMSide(game, d=D, gamma=GAMMA, lbd=LBD,
                            use_water_transfer=False)),
]

# ── Run all algorithms ─────────────────────────────────────────────────────────
all_results = {}

for name, factory in ALGOS:
    runs = []
    print(f"\nRunning {name}", end="", flush=True)

    for seed in range(N_SEEDS):
        alg = factory()
        # reset() initialises contexts for algorithms that don't do it in __init__
        # (e.g. STAP_Helmbolt).  Safe to call on all algorithms.
        alg.reset()

        # Suppress the verbose per-step prints from evaluation_contextual and
        # STAP_Helmbolt so the terminal stays readable.
        with contextlib.redirect_stdout(io.StringIO()):
            cr = evaluator.eval_policy_once(alg, game, job=(context_gen, seed))

        runs.append(cr)
        print(f" {seed}", end="", flush=True)

    runs = np.array(runs)   # shape (N_SEEDS, HORIZON)
    all_results[name] = runs
    mean_final = runs.mean(axis=0)[-1]
    print(f"  |  mean cumulative regret @ T={HORIZON}: {mean_final:.2f}")

# ── Save raw results ───────────────────────────────────────────────────────────
np.save("comparison_at_results.npy", all_results)
print("\nRaw results saved to comparison_at_results.npy")

# ── Plot ───────────────────────────────────────────────────────────────────────
t      = np.arange(1, HORIZON + 1)
COLORS = {
    "Random":          "#aaaaaa",
    "PGTS":            "#e07b54",
    "STAP-Helmbolt":   "#5b8db8",
    "SquareCB.PMSide": "#3a9668",
}
STYLES = {
    "Random":          "--",
    "PGTS":            "-.",
    "STAP-Helmbolt":   ":",
    "SquareCB.PMSide": "-",
}

fig, ax = plt.subplots(figsize=(7, 4.5))

for name, _ in ALGOS:
    runs  = all_results[name]
    mean  = runs.mean(axis=0)
    std   = runs.std(axis=0)
    color = COLORS[name]
    ls    = STYLES[name]
    ax.plot(t, mean, label=name, color=color, linewidth=1.8, linestyle=ls)
    ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=color)

ax.set_xlabel("Round $t$", fontsize=12)
ax.set_ylabel("Cumulative regret", fontsize=12)
ax.set_title("Apple Tasting — contextual algorithm comparison", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linewidth=0.35, alpha=0.6)
fig.tight_layout()
fig.savefig("comparison_at_regret.png", dpi=150)
print("Plot saved to comparison_at_regret.png")
