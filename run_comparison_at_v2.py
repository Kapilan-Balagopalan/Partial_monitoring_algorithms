"""
Full algorithm comparison on Apple Tasting (contextual setting).

Algorithms:
  - Random           — uniform baseline
  - PGTS             — Polya-Gamma Thompson Sampling
  - STAP-Helmbolt    — Helmbold et al.
  - CBPside          — Confidence Bounds for PM (deterministic)
  - RandCBPside      — Confidence Bounds for PM (randomized) [paper contribution]
  - SquareCB.PMSide  — IGW + water-transfer [this work]

Run:
    py run_comparison_at_v2.py

Output:
    comparison_at_v2_regret.png
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
from cbpside import CBPside
from randcbpside import RandCPBside

# ── Experiment settings ────────────────────────────────────────────────────────
HORIZON = 3000
N_SEEDS = 5
D       = 10         # context dimension — must be >= 5 for CBPside/RandCBPside
                     # (their confidence formula is sqrt((d-4)*log(t)); the
                     # notebook uses d=10, so we match that here)
GAMMA   = 1.0        # SquareCB.PMSide: effective rate gamma_t = GAMMA * sqrt(t)
LBD     = 0.05       # ridge regularisation
ALPHA   = 1.01       # CBPside / RandCBPside confidence-bound exponent

# 10-dim weight vector matching the notebook setup (seed 1)
np.random.seed(1)
W = np.random.uniform(0, 0.1, D)
W = W / W.sum()

# RandCBPside-specific
SIGMA   = 5          # bandwidth for the randomised confidence level
K       = 10         # number of quantisation levels
EPSILON = 10e-7      # exploration probability floor

# ── Setup ──────────────────────────────────────────────────────────────────────
game        = apple_tasting(restructure_game=False)
context_gen = LinearContexts(W)
evaluator   = Evaluation_contextual(HORIZON)

# ── Algorithm registry ─────────────────────────────────────────────────────────
ALGOS = [
    ("Random",
     lambda: Random(game, HORIZON)),

    ("PGTS",
     lambda: PGTS(game, D)),

    ("STAP-Helmbolt",
     lambda: STAP_Helmbolt(game, D)),

    ("CBPside",
     lambda: CBPside(game, D, ALPHA, LBD)),

    ("RandCBPside",
     lambda: RandCPBside(game, D, ALPHA, LBD, SIGMA, K, EPSILON)),

    ("SquareCB.PMSide",
     lambda: SquareCBPMSide(game, d=D, gamma=GAMMA, lbd=LBD,
                            use_water_transfer=False)),
]

# ── Run ────────────────────────────────────────────────────────────────────────
all_results = {}

for name, factory in ALGOS:
    runs = []
    print(f"\nRunning {name}", end="", flush=True)

    for seed in range(N_SEEDS):
        alg = factory()
        # reset() initialises contexts for algorithms that skip it in __init__
        # (STAP_Helmbolt, RandCBPside).  Safe for all others too.
        alg.reset()

        with contextlib.redirect_stdout(io.StringIO()):
            cr = evaluator.eval_policy_once(alg, game, job=(context_gen, seed))

        runs.append(cr)
        print(f" {seed}", end="", flush=True)

    runs = np.array(runs)
    all_results[name] = runs
    print(f"  |  mean final regret: {runs.mean(axis=0)[-1]:.2f}")

np.save("comparison_at_v2_results.npy", all_results)
print("\nRaw results saved to comparison_at_v2_results.npy")

# ── Plot ───────────────────────────────────────────────────────────────────────
t = np.arange(1, HORIZON + 1)

COLORS = {
    "Random":          "#aaaaaa",
    "PGTS":            "#e07b54",
    "STAP-Helmbolt":   "#5b8db8",
    "CBPside":         "#c07ad4",
    "RandCBPside":     "#d4a017",
    "SquareCB.PMSide": "#3a9668",
}
STYLES = {
    "Random":          "--",
    "PGTS":            "-.",
    "STAP-Helmbolt":   ":",
    "CBPside":         (0, (3, 1, 1, 1)),   # dash-dot-dot
    "RandCBPside":     (0, (5, 2)),          # loose dash
    "SquareCB.PMSide": "-",
}

fig, ax = plt.subplots(figsize=(8, 5))

for name, _ in ALGOS:
    runs  = all_results[name]
    mean  = runs.mean(axis=0)
    std   = runs.std(axis=0)
    ax.plot(t, mean, label=name, color=COLORS[name],
            linewidth=1.8, linestyle=STYLES[name])
    ax.fill_between(t, mean - std, mean + std, alpha=0.12, color=COLORS[name])

ax.set_xlabel("Round $t$", fontsize=12)
ax.set_ylabel("Cumulative regret", fontsize=12)
ax.set_title("Apple Tasting — contextual algorithm comparison", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linewidth=0.35, alpha=0.6)
fig.tight_layout()
fig.savefig("comparison_at_v2_regret.png", dpi=150)
print("Plot saved to comparison_at_v2_regret.png")
