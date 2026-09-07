import numpy as np
from polyagamma import random_polyagamma
from scipy.special import expit


class PGTSMultiArm():
    """Multi-arm Polya-Gamma Thompson Sampling for partial-monitoring games with binary signals.

    Model and sampler follow PG-TS (Dumitrascu, Feng & Engelhardt, NeurIPS 2018,
    Algorithm 2): a Bayesian logistic regression of the binary observation of the
    played arm, theta ~ N(b, B), sampled with the Polya-Gamma Gibbs sampler
    (Polson, Scott & Windle 2013), one posterior draw per round (Thompson sampling).
    Arms are the game's actions with per-arm features, i.e. one parameter vector
    theta_a per action; the binary observation of arm a is its feedback symbol
    (sold / not sold in dynamic pricing, revealed label in apple tasting).

    Decision rule (the partial-monitoring adaptation): the loss depends on the
    unobserved outcome, not on the observed signal, so the sampled signal
    probabilities of all arms are combined into an outcome distribution hat_q by
    least squares on the stacked signal matrices (as in SquareCBPMSide) and the arm
    minimising the expected loss  L @ hat_q  is played.  For the two-action apple
    tasting structure this is PGTS: the informative arm's logistic model and the
    argmin of the expected loss.

    Requirements: every action has at most two feedback symbols.  Actions with a
    single symbol are uninformative: their signal is known and no model is fitted.

    Parameters
    ----------
    game          : Game object
    d             : context dimension
    gibbsits      : Gibbs iterations per round (warm-started from the previous round)
    prior_mean    : b, prior mean of every coefficient (PGTS.py uses 0.5)
    prior_cov     : B = prior_cov * I
    add_intercept : append a constant feature to the per-arm features (per-arm bias)
    """

    def __init__(self, game, d, gibbsits=10, prior_mean=0.0, prior_cov=1.0, add_intercept=True):
        self.game = game
        self.N = game.n_actions
        self.M = game.n_outcomes
        self.SignalMatrices = game.SignalMatrices
        for a in range(self.N):
            if len(self.SignalMatrices[a]) > 2:
                raise ValueError("PGTSMultiArm requires binary observations (at most two feedback symbols per action)")
        self.informative = [a for a in range(self.N) if len(self.SignalMatrices[a]) == 2]

        self.gibbsits = gibbsits
        self.add_intercept = add_intercept
        self.d = d + (1 if add_intercept else 0)
        self.pmean = np.full(self.d, float(prior_mean))
        self.pcovar_inv = np.identity(self.d) / float(prior_cov)     # B^{-1}
        self.reset()

    def reset(self):
        self.contexts = {a: {'features': [], 'labels': []} for a in range(self.N)}
        self.theta = {a: self.pmean.copy() for a in range(self.N)}   # warm start of each Gibbs chain

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _features(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)
        return np.append(x, 1.0) if self.add_intercept else x

    def thetagibbs(self, a):
        """Polya-Gamma Gibbs sampler for arm a (PG-TS Algorithm 2); returns the last draw."""
        features = np.array(self.contexts[a]['features'])            # (n, d)
        kappa = np.array(self.contexts[a]['labels'], dtype=float) - 0.5
        theta = self.theta[a]
        for _ in range(self.gibbsits):
            omega = random_polyagamma(1, features @ theta)            # omega_i ~ PG(1, x_i^T theta)
            V = np.linalg.inv(features.T @ (omega[:, None] * features) + self.pcovar_inv)
            m = V @ (features.T @ kappa + self.pcovar_inv @ self.pmean)
            theta = np.random.multivariate_normal(m, V)
        self.theta[a] = theta
        return theta

    def signal_probabilities(self, X, sample=True):
        """P(symbol 1 | x, a) for every informative arm with data (posterior draw, or current theta)."""
        phi = self._features(X)
        probs = {}
        for a in self.informative:
            if not self.contexts[a]['labels']:
                continue
            theta = self.thetagibbs(a) if sample else self.theta[a]
            probs[a] = float(expit(theta @ phi))
        return probs

    def estimate_outcome_dist(self, X, sample=True):
        """hat_q: least-squares combination of the arms' predicted signal distributions."""
        probs = self.signal_probabilities(X, sample)
        rows, preds = [], []
        for a in range(self.N):
            if a in self.informative:
                if a not in probs:
                    continue
                rows.append(self.SignalMatrices[a])
                preds.append(np.array([1.0 - probs[a], probs[a]]))
            else:                                                    # single symbol: known signal
                rows.append(self.SignalMatrices[a])
                preds.append(np.ones(1))
        S_stack = np.vstack(rows)
        if np.linalg.matrix_rank(S_stack) < self.M:
            return np.ones(self.M) / self.M
        q = np.linalg.lstsq(S_stack, np.concatenate(preds), rcond=None)[0]
        q = np.clip(q, 0.0, None)
        s = q.sum()
        return q / s if s > 0 else np.ones(self.M) / self.M

    # ------------------------------------------------------------------
    # Main loop interface
    # ------------------------------------------------------------------

    def get_action(self, t, X):
        if t < self.N:                       # play each arm once
            return t
        hat_q = self.estimate_outcome_dist(X)
        expected_loss = self.game.LossMatrix @ hat_q
        return int(np.argmin(expected_loss))

    def update(self, action, feedback, outcome, t, context):
        if action not in self.informative:
            return
        e_y = np.zeros(self.M)
        e_y[outcome] = 1
        symbol = int(np.argmax(self.SignalMatrices[action] @ e_y))   # observed feedback symbol, 0 or 1
        self.contexts[action]['features'].append(self._features(context))
        self.contexts[action]['labels'].append(symbol)
