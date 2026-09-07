from math import log, exp, pow
import numpy as np
# import geometry
import collections
import geometry_v3
from scipy.optimize import fsolve

class Game():
    
    def __init__(self, name, LossMatrix, FeedbackMatrix,FeedbackMatrix_PMDMED, banditLossMatrix,  banditFeedbackMatrix, LinkMatrix, SignalMatrices, signal_matrices_Adim, mathcal_N, v, N_plus, V ,  mode = None):
        
        self.name = name
        self.LossMatrix = LossMatrix
        self.FeedbackMatrix = FeedbackMatrix
        self.FeedbackMatrix_PMDMED = FeedbackMatrix_PMDMED
        self.banditLossMatrix = banditLossMatrix
        self.banditFeedbackMatrix = banditFeedbackMatrix
        
        self.LinkMatrix = LinkMatrix
        self.SignalMatrices = SignalMatrices
        self.SignalMatricesAdim = signal_matrices_Adim
        self.n_actions = len(self.LossMatrix)
        self.n_outcomes = len(self.LossMatrix[0])
        self.mathcal_N = mathcal_N 
        self.v = v
        self.N_plus = N_plus
        self.V = V

        self.mode = mode

        self.N = len(self.LossMatrix)
        self.M = len(self.LossMatrix[0])
        self.Actions_dict = { a : "{0}".format(a) for a in range(self.N)} # Actions semantic
        self.Outcomes_dict = { a : "{0}".format(a) for a in range(self.M)} # Outcomes semantic

        self.outcome_dist = None #self.set_outcome_distribution()
        self.deltas, self.i_star = None, None #self.optimal_action(  )

    def set_outcome_distribution(self, outcome_distribution, jobid):
        self.jobid = jobid
        self.outcome_dist = outcome_distribution
        self.deltas, self.i_star = self.optimal_action(  )

    def optimal_action(self, ):
        deltas = []
        for i in range(len(self.LossMatrix)):
            deltas.append( self.LossMatrix[i,...].T @ list( self.outcome_dist.values() ) )
        return deltas, np.argmin(deltas)

    def delta(self, action):
        return ( self.LossMatrix[action,...] - self.LossMatrix[self.i_star,...] ).T @ list( self.outcome_dist.values() ) 

def apple_tasting( restructure_game ):

    name = 'AT'
    init_LossMatrix = np.array( [ [1, 0], [0, 1] ] )
    init_FeedbackMatrix =  np.array([ [1, 1],[1, 0] ])
    signal_matrices =  [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ] ) ]

    bandit_LossMatrix = np.array( [ [1, 0], [0, 1] ] )
    bandit_FeedbackMatrix =  np.array([ [0, 0],[0, -1] ])

    FeedbackMatrix_PMDMED =  np.array([ [0, 0],[1, 2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,1],[0,0],[0,0] ] ), np.array( [ [0,0],[1,0],[0,1] ] ) ]

    mathcal_N = [ [0, 1] ]

    # if restructure_game:
    #     FeedbackMatrix, LossMatrix = general_algorithm( init_FeedbackMatrix, init_LossMatrix )
    # else:
    FeedbackMatrix, LossMatrix = init_FeedbackMatrix, init_LossMatrix

    v = {0: {1: [np.array([0]), np.array([-1.,  1.])]} }
    #collections.defaultdict(dict)
    #v[0][1] = [  np.array([[0.5]]), np.array([[-1.5, 0.5]])  ]
    #v[1][0] = [  np.array([[0.5]]), np.array([[0.5, -1.5]])  ]

    N_plus =  collections.defaultdict(dict)
    N_plus[0][1] = [ 0, 1 ]

    V = collections.defaultdict(dict)
    V[0][1] = [ 0,1 ]
    
    LinkMatrix = np.linalg.inv( init_FeedbackMatrix ) @ LossMatrix 

    game = Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix, LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

    return game


def label_efficient(  ):

    name = 'LE'
    LossMatrix = np.array( [ [1, 1],[1, 0],[0, 1] ] )
    FeedbackMatrix = np.array(  [ [0, 1], [2, 2], [3, 3] ] )

    signal_matrices = [ np.array( [ [1,0],[0,1] ]), np.array( [ [1,1] ] ), np.array( [ [1,1] ] ) ] 

    bandit_LossMatrix = None
    bandit_FeedbackMatrix =  None
    LinkMatrix = None

    FeedbackMatrix_PMDMED =  np.array([ [0, 1],[2, 2],[2,2] ])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED,  len(FeedbackMatrix_PMDMED),len(FeedbackMatrix_PMDMED[0]) )
    signal_matrices_Adim =  [ np.array( [ [1,0],[0,1],[0,0] ] ), np.array( [ [0,0],[0,0],[1,1] ] ), np.array( [ [0,0],[0,0],[1,1] ] ) ]
    
    mathcal_N = [  [1,2], ] 

    v = {1: {2: [ np.array([1.,  -1.]), np.array([0]), np.array([0])]} }
    
    N_plus =  collections.defaultdict(dict)
    N_plus[1][2] = [ 1, 2 ]

    V = collections.defaultdict(dict)
    V[1][2] = [ 0, 1, 2 ]

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,  LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )


def label_efficient_modified():
    """Label Efficient game with action 0 loss changed to [0.4, 0.4].

    Cell boundaries:  q=0.4 (between a0 and a2) and q=0.6 (between a0 and a1).
    Action 0 is now the central node: neighbourhood = {(0,1), (0,2)}.

    Loss difference vectors (analytically derived, using only S_0 = I_2):
      L[0] - L[1] = [-0.6,  0.4]   -> v[0][1][0] = [-0.6,  0.4]
      L[0] - L[2] = [ 0.4, -0.6]   -> v[0][2][0] = [ 0.4, -0.6]
    Actions 1 and 2 give no signal (S_k = [[1,1]]), so their v-coefficients are 0.
    """
    name = 'LE_mod'
    LossMatrix        = np.array([[0.4, 0.4], [1, 0], [0, 1]])
    FeedbackMatrix    = np.array([[0, 1], [2, 2], [3, 3]])
    signal_matrices   = [
        np.array([[1, 0], [0, 1]]),   # action 0: fully informative (S_0 = I_2)
        np.array([[1, 1]]),            # action 1: uninformative
        np.array([[1, 1]]),            # action 2: uninformative
    ]

    bandit_LossMatrix    = None
    bandit_FeedbackMatrix = None
    LinkMatrix           = None

    FeedbackMatrix_PMDMED = np.array([[0, 1], [2, 2], [2, 2]])
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED, 3, 2)
    signal_matrices_Adim = [
        np.array([[1, 0], [0, 1], [0, 0]]),
        np.array([[0, 0], [0, 0], [1, 1]]),
        np.array([[0, 0], [0, 0], [1, 1]]),
    ]

    mathcal_N = [[0, 1], [0, 2]]

    # v[i][j] is a list indexed by action (same convention as label_efficient).
    # Only action 0's coefficient is non-zero for both pairs.
    v = {0: {
        1: [np.array([-0.6,  0.4]), np.array([0.]), np.array([0.])],
        2: [np.array([ 0.4, -0.6]), np.array([0.]), np.array([0.])],
    }}

    N_plus = collections.defaultdict(dict)
    N_plus[0][1] = [0, 1]
    N_plus[0][2] = [0, 2]

    V = collections.defaultdict(dict)
    V[0][1] = [0, 1, 2]
    V[0][2] = [0, 1, 2]

    return Game(name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED,
                bandit_LossMatrix, bandit_FeedbackMatrix, LinkMatrix,
                signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V)



def dynamic_pricing(n_prices=2, c=2.0, n_valuations=None):
    """Dynamic pricing game (Kleinberg & Leighton 2003; Bartok, Pal & Szepesvari 2011).

    A seller posts one of N prices {1, ..., N}, the buyer has a hidden valuation in
    {1, ..., M} (M = N by default); a sale happens iff price <= valuation.
        loss     = valuation - price  if sold (lost revenue),   c  if not sold
        feedback = sold / not sold
    This is the benchmark game of the CBP / PM-DMED / TSPM experiments (N = M = 5, c = 2).

    Conventions (chosen so that N = M = 2 has the structure of apple_tasting()):
        action  i  <->  price      i + 1     (increasing)
        outcome j  <->  valuation  M - j     (decreasing: outcome 0 = highest valuation)
    For N = M = 2:  LossMatrix = [[1, 0], [0, c]],  FeedbackMatrix = [[1, 1], [1, 0]],
    the low price always sells (uninformative) and the high price reveals the valuation;
    with c = 1 the game is exactly apple tasting.  For M >= 3 no single price reveals
    the valuation (globally but not locally observable game).

    The geometry used by CBPside / RandCBPside is computed with geometry_v3 (Gurobi):
    Pareto-optimal actions, neighbouring pairs mathcal_N, neighbourhood action sets
    N_plus, observer sets V = all actions, observer vectors v (minimum-norm solution
    of  sum_k S_k^T v_k = L_i - L_j  over the informative actions, zeros elsewhere,
    as in the hand-written games).
    """
    N = int(n_prices)
    M = N if n_valuations is None else int(n_valuations)
    prices = np.arange(1, N + 1, dtype=float)
    valuations = np.arange(M, 0, -1, dtype=float)          # outcome 0 = highest valuation
    sold = prices[:, None] <= valuations[None, :]           # (N, M)
    LossMatrix = np.where(sold, valuations[None, :] - prices[:, None], float(c))
    FeedbackMatrix = sold.astype(int)                       # 1 = sold, 0 = not sold

    # Signal matrices: one row per distinct feedback symbol of the action (symbol order 0, 1).
    signal_matrices = []
    for i in range(N):
        symbols = sorted(set(FeedbackMatrix[i]))
        signal_matrices.append(np.array([[1 if FeedbackMatrix[i][j] == sym else 0 for j in range(M)]
                                         for sym in symbols]))

    # Feedback matrix with globally unique symbols (PM-DMED / TSPM convention, cf. apple_tasting).
    FeedbackMatrix_PMDMED = np.zeros((N, M), dtype=int)
    next_symbol = 0
    for i in range(N):
        seen = {}
        for j in range(M):
            f = FeedbackMatrix[i][j]
            if f not in seen:
                seen[f] = next_symbol
                next_symbol += 1
            FeedbackMatrix_PMDMED[i][j] = seen[f]
    A = geometry_v3.alphabet_size(FeedbackMatrix_PMDMED, N, M)
    signal_matrices_Adim = geometry_v3.calculate_signal_matrices(FeedbackMatrix_PMDMED, N, M, A)

    bandit_LossMatrix = LossMatrix.copy()
    bandit_FeedbackMatrix = None
    LinkMatrix = None
    if N == M and abs(np.linalg.det(FeedbackMatrix)) > 1e-12:
        LinkMatrix = np.linalg.inv(FeedbackMatrix) @ LossMatrix

    # Geometry (Pareto-optimal actions, neighbours, neighbourhood action sets, observer vectors).
    pareto = geometry_v3.getParetoOptimalActions(LossMatrix, N, M, [])
    mathcal_N = [[i, j] for a, i in enumerate(pareto) for j in pareto[a + 1:]
                 if geometry_v3.isNeighbor(LossMatrix, N, M, i, j, [])]

    informative = [k for k in range(N) if len(signal_matrices[k]) > 1]
    N_plus = collections.defaultdict(dict)
    V = collections.defaultdict(dict)
    V_informative = collections.defaultdict(dict)
    for i, j in mathcal_N:
        N_plus[i][j] = geometry_v3.getNeighborhoodActionSet(LossMatrix, N, M, i, j)
        V[i][j] = list(range(N))
        V_informative[i][j] = list(informative)
    v_informative = geometry_v3.getV(LossMatrix, N, M, FeedbackMatrix, signal_matrices, mathcal_N, V_informative)
    v = {}
    for i, j in mathcal_N:
        v.setdefault(i, {})[j] = [v_informative[i][j][k] if k in informative else np.zeros(len(signal_matrices[k]))
                                  for k in range(N)]

    return Game('DP', LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,
                LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V)


################### tau detection game:


def objective_fn(b, a, T):
    return a/b - T

def solve_system(a, T):
    def objective(b):
        return objective_fn(b, a, T)

    b_opt = fsolve(objective, x0=1.0)
    return b_opt

def label_efficient2( threshold ): ### \tau detection game (in the paper)

    name = 'LE2'
    a = 1
    b_opt = int( np.round( solve_system(a, threshold)[0] ) )
    LossMatrix = np.array( [ [a,a], [b_opt, 0] ] )
    FeedbackMatrix = np.array(  [ [1, 0], [1, 1]  ] )
    signal_matrices = [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ])  ] 

    bandit_LossMatrix = None
    bandit_FeedbackMatrix =  None

    FeedbackMatrix_PMDMED =  None
    A = None
    signal_matrices_Adim =  None

    mathcal_N = [  [0, 1],  [1, 0] ] 

    V = collections.defaultdict(dict)
    V[1][0] = [ 0, 1 ]
    V[0][1] = [ 0, 1 ]

    N_plus =  collections.defaultdict(dict)
    N_plus[1][0] = [ 0, 1 ]
    N_plus[0][1] = [ 1, 0 ]

    LossMatrix = np.array( [ [1, 0], [0.5, 0.5] ] )
    FeedbackMatrix = np.array(  [ [1, 0], [1, 1]  ] )
    LinkMatrix = None
    signal_matrices = [ np.array( [ [1,1] ] ), np.array( [ [0,1], [1,0] ])  ] 

    bandit_LossMatrix = None
    bandit_FeedbackMatrix =  None

    FeedbackMatrix_PMDMED =  None
    A = None
    signal_matrices_Adim =  None
    
    mathcal_N = [  [0, 1],  [1, 0] ] 

    v = {0: {1: {0: np.array([ 0.5, -0.5]), 1: np.array([0.])}},
             1: {0: {0: np.array([-0.5,  0.5]), 1: np.array([0.])}}} 

    N_plus =  collections.defaultdict(dict)
    N_plus[1][0] = [ 0, 1 ]
    N_plus[0][1] = [ 1, 0 ]

    V = collections.defaultdict(dict)
    V[1][0] = [ 0, 1 ]
    V[0][1] = [ 0, 1 ]

    v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,  LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )

