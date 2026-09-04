
# from signal import valid_signals
import geometry_v3
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import scipy
import cyipopt
import tnlp


class PM_DMED:

    def __init__(self, game, horizon, c):

      self.game = game
      self.horizon = horizon
      self.N = game.n_actions
      self.M = game.n_outcomes
    
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix_PMDMED, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M, 'alphabet', self.A)

      self.ZC = set( [ i for i in range(self.N) ] )
      self.ZR = set(  [ i for i in range(self.N) ] )
      self.ZRc = set([])
      self.ZN = set([])

      self.constraintNum = 100
      self.VERY_LARGE_DOUBLE = 10000000
      self.VERY_SMALL_DOUBLE  = 0.000001

      self.c = c

      self.feedback = [] 
      for _ in range(self.N):
          self.feedback.append(  np.ones( self.A ) ) # ones and not zeros for avoiding infeasible models

    def get_action(self, t, context = None):
    #   print('current set', self.ZC)
      action = self.ZC.pop()
      return action 

    def reset(self, ):
      self.ZC = set( [ i for i in range(self.N) ] )
      self.ZR = set( [ i for i in range(self.N) ] )
      self.ZRc = set( [] )
      self.ZN = set( [] )
      self.feedback = [] 
      for _ in range(self.N):
          self.feedback.append(  np.ones( self.A ) ) # ones and not zeros for avoiding infeasible models

    def update(self, action, feedback, outcome, context, t):
 
      feedback = self.game.FeedbackMatrix_PMDMED[action][outcome]
      self.feedback[action][ feedback ] += 1

    #   self.ZC.remove(action)
      self.ZRc.add(action)

      if self.use_cache(t):
          suffExplore = self.suffExplore_cache.copy()
      else:
          p = self.MLE( 0.1/np.sqrt(t+1) )
          self.suffExplore_cache = self.sufficientExploration( p , np.max( [np.log(t+1), 1.0] )  )
          suffExplore = self.suffExplore_cache

    #   p = self.MLE( 0.1/np.sqrt(t+1) )
    #   suffExplore = self.sufficientExploration( p , np.max( [np.log(t+1), 1.0] )  )
      ins_actions = []
      for i in range(self.N):
          if suffExplore[i] >= sum(self.feedback[i]):
            #   print('ajout action ',i, ' seuil ',suffExplore[i], ' reality ',  sum(self.feedback[i]) )
              ins_actions.append(i)
          
      for i in self.ZRc:
          if i in ins_actions:
            #   print('ajout action par ins_actions', i)
              self.ZN.add(i)

      for i in range(self.N):
        if i in self.ZRc and sum(self.feedback[i]) < np.sqrt(np.log(t+1)):
          self.ZN.add(i)


      if not self.ZC:  # check if LC is empty
        self.ZC = self.ZN.copy()
        self.ZRc = set(range(self.N)) - self.ZC  # create complementary set
        self.ZN = set([])

    def use_cache(self, t):
      return (t>100) and ( (t%10)!=0 )

    def MLE(self, accuracy): #estimate the outcome distribution

      x_lb = [ 0.0001 for _ in range(self.M)]
      x_ub = [ 0.999 for _ in range(self.M)]
      gl = [1.0]
      gu = [1.0]
      nlp = cyipopt.Problem( n=self.M, m=1,
                             problem_obj=tnlp.MLE_NLP(self.game, self.A, self.feedback ),
                             lb=x_lb, ub=x_ub, cl=gl, cu=gu )
      accuracy = 0.1
      cyipopt.set_logging_level(0)
      nlp.add_option('sb', 'yes')
      nlp.add_option('print_level', 0)
      nlp.add_option('mu_strategy', 'adaptive')
      nlp.add_option("tol", 0.1 * accuracy) # 0.1 * accuracy
      nlp.add_option("acceptable_tol", 1 * accuracy)
      nlp.add_option("bound_relax_factor", 0.0)
      nlp.add_option("jac_c_constant", "yes")
      nlp.add_option("hessian_approximation", "limited-memory")
      x0 = [ 1/self.M for _ in range(self.M) ]
      try:
        x, info = nlp.solve(x0)
        if info['status'] == 0 :
          solution = x
        else:
        #   print('erreur optimisation 1')
          solution = np.random.uniform(0, 1, self.M)
          solution = solution / solution.sum()
      except TypeError:
        # print('erreur optimisation 2')
        solution = np.random.uniform(0, 1, self.M)
        solution = solution / solution.sum()
      #print('x',x)
      #print('info', info)
      return solution

##########################################################################################
##########################################################################################
##########################################################################################

    def sufficientExploration(self, p, logt ):

      i_star = self.getOptimalAction(p)

      ps = [ self.getConstrainedRandomPoint( p, i_star ) for _ in range(self.constraintNum) ]
    #   print('ps', ps)

      m = gp.Model( )
      m.Params.LogToConsole = 0

      Deltas = np.zeros(self.N)
      numZeros = 0
      for i in range(self.N):
        ldiff = self.game.LossMatrix[i,...] - self.game.LossMatrix[i_star, ...]
        Deltas[i] +=  ldiff.T @ p
        if Deltas[i] < self.VERY_SMALL_DOUBLE:
          numZeros += 1
    #   print('deltas', Deltas)

      #print('hey')
      
      if numZeros>=2: #degenerate optimal solution
        # print('degenerate optimal solution')
        vals = np.zeros(self.N)
        for i in range(self.N):
          if Deltas[i] < self.VERY_SMALL_DOUBLE:
            vals[i] = self.VERY_LARGE_DOUBLE
        return vals 

      vars = { i:0 for i in range(self.N) }
      for i in range(self.N):
        if i != i_star: 
          varName = "N_{}".format(i) 
          vars[i] =  m.addVar(0, GRB.INFINITY, Deltas[i], GRB.CONTINUOUS, varName ) 
          m.update()

      #print('hola')
      #print('vars', vars)
      for _ in range( len(ps) ):
        if self.getOptimalAction( ps[_] ) != i_star:
          ps[_] = self.findBoundaryByBisection( p, ps[_], i_star)
          ConstExpr = 0
          for i in range(self.N):
            if i != i_star:
              di = self.kl_i(p, ps[_], i)
              ConstExpr += di * vars[i]
          m.addConstr( ConstExpr >= logt,  'constraint{}'.format(_) ) # may need to be removed ?
      
      #print('heyyy')
      
      m.optimize()
      # print(m.status )
      # optimal action should be played infinite number of times:
      if m.status == 2:
        vals = [ self.VERY_LARGE_DOUBLE if i == i_star else vars[i].X for i in range(self.N) ]
        vals = np.array(vals)
      else:
        # print('s')
        vals = [ self.VERY_LARGE_DOUBLE if i == i_star else self.VERY_LARGE_DOUBLE for i in range(self.N) ]
        vals = np.array(vals)

      
      
    #   vals = np.zeros(self.N)
    #   for i in range(self.N):
    #       if i == i_star:
    #         vals[i] = self.VERY_LARGE_DOUBLE
    #       else:
    #           try:
    #             vars[i].X
    #           except AttributeError:
    #             vals[i] = self.VERY_LARGE_DOUBLE


 
    #   print(vals)

      return vals

    def getOptimalAction(self, p):
      deltas = []
      for i in range( len(self.game.LossMatrix) ):
        deltas.append( self.game.LossMatrix[i,...].T @ p )
      return np.argmin(deltas)

    def getConstrainedRandomPoint(self, p, i_star):
      pSample = np.zeros( self.M )
      for a in range(self.A):
        psum = 0
        sum = 0
        for j in range(self.M):
          if self.game.SignalMatricesAdim[i_star][a][j]: 
            pSample[j] = np.random.uniform(0,1)
            psum += p[j]
            sum += pSample[j]
        for j in range(self.M):
          if self.game.SignalMatricesAdim[i_star][ a ][j]: 
            pSample[j] *= psum/sum
      return pSample

    def findBoundaryByBisection(self, p, p2, i_star):
      lb = 0
      ub = 1
      while ( (ub-lb) > 0.01 ):
        c = (ub+lb) / 2
        pm = (1-c) * p + c * p2
        if self.getOptimalAction( pm ) == i_star:
          lb = c
        else:
          ub = c
      return (1-lb) * p + lb * p2

    def kl_i(self, p1, p2, i):
      p1_sym = np.zeros(self.A)
      p2_sym = np.zeros(self.A)
      for j in range(self.M):
        p1_sym[  self.game.FeedbackMatrix_PMDMED[i,j] ] += p1[j]
        p2_sym[  self.game.FeedbackMatrix_PMDMED[i,j] ] += p2[j]
      return scipy.special.kl_div( p1_sym , p2_sym ).sum()
    


