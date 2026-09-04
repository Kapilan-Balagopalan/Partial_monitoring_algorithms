"""
Run SquareCBPMSide on the Apple Tasting game (contextual setting).
Execute from the repo root:

    python run_squarecb_at.py

Results are printed to stdout and saved to squarecb_at_results.npy.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed; saves a PNG
import matplotlib.pyplot as plt

from games import apple_tasting
from squarecb_pmside import SquareCBPMSide
from evaluation_contextual import Evaluation_contextual
from synthetic_data import LinearContexts

# ── Experiment settings ────────────────────────────────────────────────────────
HORIZON    = 3000
N_SEEDS    = 5
D          = 2       # context dimension (must match LinearContexts)

# Algorithm hyper-parameters
# gamma_t = GAMMA * sqrt(t) gives O(sqrt(T)) total regret.
# Base rate GAMMA=1 means at t=3000 the effective rate is ~55,
# giving p_igw[wrong] ~ 1/(55 * gap) per round for large t.
# mu is set internally to N (number of actions) — not a free parameter.
GAMMA      = 1.0
LBD        = 0.05    # ridge regularisation

# True outcome-probability weight vector  (p_spam = w @ context)
W          = np.array([0.6, 0.4])

USE_WATER_TRANSFER = False   # identity is sufficient for 2-action Apple Tasting

# ── Setup ──────────────────────────────────────────────────────────────────────
game          = apple_tasting(restructure_game=False)
context_gen   = LinearContexts(W)
evaluator     = Evaluation_contextual(HORIZON)

print(f"Game           : Apple Tasting  ({game.n_actions} actions, {game.n_outcomes} outcomes)")
print(f"Horizon        : {HORIZON}")
print(f"Seeds          : {N_SEEDS}")
print(f"Water transfer : {USE_WATER_TRANSFER}")
print(f"Informative action for hat_q: ", end="")

# ── Run ────────────────────────────────────────────────────────────────────────
all_regrets = []

for seed in range(N_SEEDS):
    alg = SquareCBPMSide(
        game,
        d=D,
        gamma=GAMMA,
        lbd=LBD,
        use_water_transfer=USE_WATER_TRANSFER,
    )
    if seed == 0:
        print(alg.informative_action)   # printed once

    cumregret = evaluator.eval_policy_once(
        alg, game, job=(context_gen, seed)
    )
    all_regrets.append(cumregret)
    print(f"  seed {seed:2d}  |  cumulative regret at T={HORIZON}: {cumregret[-1]:.4f}")

all_regrets = np.array(all_regrets)   # shape (N_SEEDS, HORIZON)
mean_regret = all_regrets.mean(axis=0)
std_regret  = all_regrets.std(axis=0)

# ── Save raw results ───────────────────────────────────────────────────────────
np.save("squarecb_at_results.npy", all_regrets)
print(f"\nRaw results saved to squarecb_at_results.npy")

# ── Plot ───────────────────────────────────────────────────────────────────────
t = np.arange(1, HORIZON + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, mean_regret, label=r"SquareCBPMSide ($\gamma_t = \gamma\sqrt{t}$)")
ax.fill_between(
    t,
    mean_regret - std_regret,
    mean_regret + std_regret,
    alpha=0.2,
)
ax.set_xlabel("Round $t$")
ax.set_ylabel("Cumulative regret")
ax.set_title("SquareCB.PMSide — Apple Tasting")
ax.legend()
ax.grid(True, linewidth=0.4)
fig.tight_layout()
fig.savefig("squarecb_at_regret.png", dpi=150)
print("Plot saved to squarecb_at_regret.png")
