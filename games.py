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

