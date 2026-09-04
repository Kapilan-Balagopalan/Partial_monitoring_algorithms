import heapq
import numpy as np
import geometry_v3


class SquareCBPMSide():
    """
    SquareCB.PMSide algorithm for contextual partial monitoring.

    Parameters
    ----------
    game              : Game object
    d                 : context dimension
    gamma             : learning rate for IGW (larger = more exploitation)
    mu                : exploration floor for IGW (prevents division by zero)
    lbd               : ridge regression regularization
    use_water_transfer: if False, use the identity operator (p_t = p^IGW_t).
                        Safe for 2-action games (e.g. Apple Tasting) since the
                        identity satisfies all three lemma conditions there.

    Note: mu (IGW exploration floor) is fixed to N (number of actions) as
    required by the lemma — ensures p^IGW[a] <= 1/N for all a != b_t,
    so b_t always gets at least 1/N probability.
    """

    def __init__(self, game, d, gamma, lbd, use_water_transfer=True):

        self.game = game
        self.d = d
        self.gamma = gamma
        self.lbd = lbd

        self.N = game.n_actions
        self.mu = self.N   # exploration floor = number of actions (fixed by lemma)
        self.M = game.n_outcomes
        self.SignalMatrices = game.SignalMatrices

        # Identify action whose signal matrix is square and invertible,
        # so we can recover the outcome distribution from signal observations.
        self.use_water_transfer = use_water_transfer

        self.informative_action = self._find_informative_action()
        self.S_inv = np.linalg.inv(self.SignalMatrices[self.informative_action])

        self.contexts = self._init_contexts()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _find_informative_action(self):
        for k in range(self.N):
            S = self.SignalMatrices[k]
            if S.shape[0] == self.M:
                try:
                    np.linalg.inv(S)
                    return k
                except np.linalg.LinAlgError:
                    continue
        raise ValueError(
            "No action with a square, invertible signal matrix found. "
            "Cannot recover the outcome distribution from signals."
        )

    def _init_contexts(self):
        return [
            {
                'features': [],
                'labels': [],
                'weights': None,
                'V_it_inv': self.lbd * np.identity(self.d),
            }
            for _ in range(self.N)
        ]

    def reset(self):
        self.contexts = self._init_contexts()

    # ------------------------------------------------------------------
    # Outcome distribution estimation  (hat_q_t)
    # ------------------------------------------------------------------

    def estimate_outcome_dist(self, X):
        """
        hat_q(x_t) = S_k^{-1} @ weights_k @ x_t
        where k is the informative action.
        Falls back to uniform when no data yet.
        """
        k = self.informative_action
        if self.contexts[k]['weights'] is None:
            return np.ones(self.M) / self.M

        signal_est = self.contexts[k]['weights'] @ X   # (M, 1)
        q = self.S_inv @ signal_est.flatten()           # (M,)
        q = np.clip(q, 0.0, None)
        s = q.sum()
        return q / s if s > 0 else np.ones(self.M) / self.M

    # ------------------------------------------------------------------
    # IGW distribution  (p^IGW_t)
    # ------------------------------------------------------------------

    def igw_distribution(self, hat_q, t):
        """
        Inverse-gap weighting with time-varying learning rate gamma_t = gamma * sqrt(t):
            p^IGW_a = 1 / (mu + gamma_t * (hat_l_a - hat_l_{b_t}))  for a != b_t
            p^IGW_{b_t} = 1 - sum_{a != b_t} p^IGW_a

        Using gamma_t = gamma * sqrt(t) gives O(sqrt(T)) total regret,
        since per-round error ~ 1/(gamma*sqrt(t)*gap) which sums to O(sqrt(T)).
        """
        hat_l = self.game.LossMatrix @ hat_q   # (N,)
        b_t = int(np.argmin(hat_l))

        gamma_t = self.gamma * np.sqrt(t)

        p = np.zeros(self.N)
        for a in range(self.N):
            if a != b_t:
                gap = hat_l[a] - hat_l[b_t]
                p[a] = 1.0 / (self.mu + gamma_t * gap)

        p_bt = 1.0 - p.sum()
        if p_bt < 0.0:
            p[b_t] = 0.0
            p /= p.sum()
        else:
            p[b_t] = p_bt

        return p, b_t, hat_l

    # ------------------------------------------------------------------
    # Water-transfer operator  W_{hat_q}(p^IGW)
    # ------------------------------------------------------------------

    def water_transfer(self, p_igw, b_t, hat_l):
        """
        Adaptive water-transfer operator rooted at b_t.

        Builds a directed in-tree over the neighbourhood graph mathcal_N
        using a loss-aware priority queue: among unvisited neighbours of the
        current tree, always add the node with the LOWEST estimated loss next,
        connecting it to the already-connected neighbour with the lowest loss
        (using hat_l[parent] as a tie-breaker).

        This guarantees  hat_l[parent] <= hat_l[child]  for every tree edge,
        which is exactly condition (a) of Lattimore (2020) Lemma 13:
          (W_q - p)^T L nu <= 0  (expected loss does not increase).

        Conditions (b) and (c) follow from the depth-based weight formula:
          W[v] = sum_{u: v in anc(u)} p_igw[u] / depth[u]

        Special cases recovered automatically:
          - 2-action AT  : star of depths 1, 2.
          - LE_mod b_t=0 : star of depths 1, 2, 2.
          - LE_mod b_t=1 : path 1->0->2 of depths 1, 2, 3
                           (since hat_l[0]=0.4 < hat_l[2]=q1 whenever b_t=1).
          - LE_mod b_t=2 : path 2->0->1 of depths 1, 2, 3.
        """
        # Build undirected adjacency from the neighbourhood graph.
        adj = [[] for _ in range(self.N)]
        for pair in self.game.mathcal_N:
            adj[pair[0]].append(pair[1])
            adj[pair[1]].append(pair[0])

        # Priority-queue spanning tree rooted at b_t.
        # Heap entries: (hat_l[candidate], hat_l[parent], candidate, parent)
        # Primary key  = loss of candidate  → lower-loss nodes added first
        # Secondary key = loss of parent    → when the same candidate is
        #                 reachable via two parents, use the lower-loss one.
        depth  = [-1] * self.N
        parent = [-1] * self.N
        depth[b_t] = 1
        in_tree = {b_t}

        heap = [(hat_l[nb], hat_l[b_t], nb, b_t) for nb in adj[b_t]]
        heapq.heapify(heap)

        while heap:
            _, _, nb, par = heapq.heappop(heap)
            if nb in in_tree:
                continue
            in_tree.add(nb)
            depth[nb]  = depth[par] + 1
            parent[nb] = par
            for nb2 in adj[nb]:
                if nb2 not in in_tree:
                    heapq.heappush(heap, (hat_l[nb2], hat_l[nb], nb2, nb))

        # Nodes unreachable from b_t via 𝒩 get depth N+1 (small positive share,
        # avoids division by zero).
        for i in range(self.N):
            if depth[i] == -1:
                depth[i] = self.N + 1

        # W[v] = sum_{u: v in anc(u)} p_igw[u] / depth[u]
        W = np.zeros(self.N)
        for u in range(self.N):
            contrib = p_igw[u] / depth[u]
            v = u
            while v != -1:
                W[v] += contrib
                v = parent[v]

        return W

    # ------------------------------------------------------------------
    # Main loop interface
    # ------------------------------------------------------------------

    def get_action(self, t, X):
        # Initialization: cycle through each action once
        if t < self.N:
            return t

        # Step 1 — estimate outcome distribution
        hat_q = self.estimate_outcome_dist(X)

        # Step 2 — IGW distribution (hat_l returned to avoid recomputing)
        p_igw, b_t, hat_l = self.igw_distribution(hat_q, t)

        # Step 3 — water-transfer (or identity for 2-action games)
        p_t = self.water_transfer(p_igw, b_t, hat_l) if self.use_water_transfer else p_igw.copy()

        # Numerical safety before sampling
        p_t = np.maximum(p_t, 0.0)
        p_t /= p_t.sum()

        return int(np.random.choice(self.N, p=p_t))

    def update(self, action, feedback, outcome, t, X):
        """Ridge regression update (same as CBPside)."""
        e_y = np.zeros((self.M, 1))
        e_y[outcome] = 1
        Y_t = self.game.SignalMatrices[action] @ e_y   # (sigma_i, 1)

        self.contexts[action]['labels'].append(Y_t)
        self.contexts[action]['features'].append(X)

        Y_it = np.squeeze(np.array(self.contexts[action]['labels']), 2).T  # (sigma_i, n)
        X_it = np.squeeze(np.array(self.contexts[action]['features']), 2).T  # (d, n)

        V_it_inv = self.contexts[action]['V_it_inv']
        low = 1 + X.T @ V_it_inv @ X
        high = V_it_inv @ X @ X.T @ V_it_inv
        self.contexts[action]['V_it_inv'] = V_it_inv - high / low
        self.contexts[action]['weights'] = Y_it @ X_it.T @ self.contexts[action]['V_it_inv']
