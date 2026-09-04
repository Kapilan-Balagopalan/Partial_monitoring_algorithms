"""
Algorithm comparison on the Modified Label Efficient game (contextual setting).

Game: label_efficient_modified()
  - Action 0 loss = [0.4, 0.4]  (changed from [1,1])
  - Observability: LOCALLY observable (action 0 is the only informative action
    and it appears in every neighbouring pair)
  - Neighbourhood: {(0,1), (0,2)} — action 0 is the central node

Algorithms included:
  - Random           — uniform baseline
  - CBPside          — reads game geometry from game object; compatible
  - RandCBPside      — reads game geometry from game object; compatible
  - SquareCB.PMSide  — general BFS water-transfer; compatible

Excluded (hardcoded to Apple Tasting / 2-action structure):
  - STAP_Helmbolt    — contexts[1] hardcoded; only supports N=2
  - PGTS             — rewfunc hardcoded if/else for actions 0 and 1 only

Run:
    py run_comparison_le_mod.py

Output:
    comparison_le_mod_regret.png
"""

import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from games import label_efficient_modified
from synthetic_data import LinearContexts
from evaluation_contextual import Evaluation_contextual
from squarecb_pmside import SquareCBPMSide
from cbpside import CBPside
from randcbpside import RandCPBside
from random_algo import Random

# ── Experiment settings ────────────────────────────────────────────────────────
HORIZON = 3000
N_SEEDS = 5
D       = 10         # context dimension (must be >= 5 for CBPside/RandCBPside)
GAMMA   = 1.0
LBD     = 0.05
ALPHA   = 1.01
SIGMA   = 5
K       = 10
EPSILON = 10e-7

# 10-dim weight vector (same seed as comparison_at_v2 for reproducibility)
np.random.seed(1)
W = np.random.uniform(0, 0.1, D)
W = W / W.sum()

# ── Setup ──────────────────────────────────────────────────────────────────────
game        = label_efficient_modified()
context_gen = LinearContexts(W)
evaluator   = Evaluation_contextual(HORIZON)

print(f"Game           : {game.name}  ({game.n_actions} actions, {game.n_outcomes} outcomes)")
print(f"Loss matrix    :\n{game.LossMatrix}")
print(f"Neighbourhood  : {game.mathcal_N}")
print(f"Observability  : locally observable")

# ── Algorithm registry ─────────────────────────────────────────────────────────
ALGOS = [
    ("Random",
     lambda: Random(game, HORIZON)),

    ("CBPside",
     lambda: CBPside(game, D, ALPHA, LBD)),

    ("RandCBPside",
     lambda: RandCPBside(game, D, ALPHA, LBD, SIGMA, K, EPSILON)),

    ("SquareCB.PMSide",
     lambda: SquareCBPMSide(game, d=D, gamma=GAMMA, lbd=LBD,
                            use_water_transfer=True)),   # BFS tree, not star
]

# ── Run ────────────────────────────────────────────────────────────────────────
all_results = {}

for name, factory in ALGOS:
    runs = []
    print(f"\nRunning {name}", end="", flush=True)

    for seed in range(N_SEEDS):
        alg = factory()
        alg.reset()
        with contextlib.redirect_stdout(io.StringIO()):
            cr = evaluator.eval_policy_once(alg, game, job=(context_gen, seed))
        runs.append(cr)
        print(f" {seed}", end="", flush=True)

    runs = np.array(runs)
    all_results[name] = runs
    print(f"  |  mean final regret: {runs.mean(axis=0)[-1]:.2f}")

np.save("comparison_le_mod_results.npy", all_results)
print("\nRaw results saved to comparison_le_mod_results.npy")

# ── Plot ───────────────────────────────────────────────────────────────────────
t = np.arange(1, HORIZON + 1)

COLORS = {
    "Random":          "#aaaaaa",
    "CBPside":         "#c07ad4",
    "RandCBPside":     "#d4a017",
    "SquareCB.PMSide": "#3a9668",
}
STYLES = {
    "Random":          "--",
    "CBPside":         (0, (3, 1, 1, 1)),
    "RandCBPside":     (0, (5, 2)),
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
ax.set_title("Modified Label Efficient — locally observable", fontsize=13)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, linewidth=0.35, alpha=0.6)
fig.tight_layout()
fig.savefig("comparison_le_mod_regret.png", dpi=150)
print("Plot saved to comparison_le_mod_regret.png")
