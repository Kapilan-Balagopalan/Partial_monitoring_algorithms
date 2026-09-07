# Partial Monitoring Algorithms

Implementation of contextual partial monitoring algorithms, extending the codebase from Heuillet et al. (ICML 2024) with additional algorithms and game variants.

## Algorithms

- **CBPside** — Confidence Bounds for Partial Monitoring (deterministic)
- **RandCBPside** — Randomized Confidence Bounds for Partial Monitoring
- **STAP-Helmbolt** — Helmbold et al. exploration strategy
- **PGTS** — Polya-Gamma Thompson Sampling
- **SquareCB.PMSide** — Inverse Gap Weighting with adaptive loss-aware water-transfer operator (added in this fork)

## Games

- **Apple Tasting** — 2-action locally observable game
- **Label Efficient** — 3-action globally observable game
- **Label Efficient (modified)** — 3-action locally observable variant with action 0 loss [0.4, 0.4]; neighbourhood {(0,1), (0,2)}
- **Dynamic Pricing** — `dynamic_pricing(n_prices, c)`: N-action partial monitoring game; no-sale cost `c` (default 2); `N = 2` is a 2-action locally observable game (Apple Tasting when `c = 1`); `N >= 3` is globally observable.

## Requirements

- Python 3.8+
- pip

```bash
pip install -r requirements.txt
```

Optional: `gurobipy` (for CBPside/RandCBPside LP solving), `polyagamma` (for PGTS).

## Running Experiments

**Apple Tasting — all algorithms:**
```bash
py run_comparison_at_v2.py
```

**Modified Label Efficient — generic algorithms:**
```bash
py run_comparison_le_mod.py
```

**SquareCB.PMSide on Apple Tasting only:**
```bash
py run_squarecb_at.py
```

**Jupyter notebooks (original experiments):**
- `experiment_contextual.ipynb` — contextual setting
- `experiment_noncontextual.ipynb` — non-contextual setting

**Jupyter notebooks (added in this fork, one experiment per notebook):**
- `experiment_contextual_dynamic_pricing.ipynb` — contextual dynamic pricing (`c = 2`) for several numbers of prices (`N_PRICES_LIST`), same protocol and algorithms as `run_comparison_at_v2.py` (PGTS and STAP only for `N = 2`); results are written to `results/`

## Acknowledgements

This codebase builds on the work of Heuillet et al.:

> *Randomized Confidence Bounds for Partial Monitoring*, ICML 2024
> https://raw.githubusercontent.com/mlresearch/v235/main/assets/heuillet24a/heuillet24a.pdf

We thank the original authors for making their code available.
