from math import log, exp, pow
import numpy as np
# import geometry
import collections
import geometry_v3
from itertools import combinations, permutations
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



def objective_fn(b, a, T):
    return a/b - T

def solve_system(a, T):
    def objective(b):
        return objective_fn(b, a, T)

    b_opt = fsolve(objective, x0=1.0)
    return b_opt

def label_efficient2( threshold ):

    name = 'LE2'
    a = 1
    b_opt = int( np.round( solve_system(a, threshold)[0] ) )
    LossMatrix = np.array( [ [a,a], [b_opt, 0] ] )
    FeedbackMatrix = np.array(  [ [1, 0], [1, 1]  ] )
    signal_matrices = [ np.array( [ [0,1], [1,0] ]), np.array( [ [1,1] ] )  ] 

    LinkMatrix = None
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

    # v = geometry_v3.getV(LossMatrix, 2, 2, FeedbackMatrix, signal_matrices, mathcal_N, V)

    v = {0: {1: [ np.array([1.,  -(b_opt - 1)]), np.array([0]) ]}, 
         1: {0: [ np.array([-1,  (b_opt - 1) ]), np.array([0]) ]}}

    return Game( name, LossMatrix, FeedbackMatrix, FeedbackMatrix_PMDMED, bandit_LossMatrix, bandit_FeedbackMatrix,  LinkMatrix, signal_matrices, signal_matrices_Adim, mathcal_N, v, N_plus, V )
