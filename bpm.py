import geometry_v3
import numpy as np


class BPM:

    def __init__(self, game, horizon,  ):
      self.game = game
      self.horizon = horizon

      self.N = game.n_actions
      self.M = game.n_outcomes
      # self.A = geometry_v3.alphabet_size(game.FeedbackMatrix, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M)

      self.SignalMatrices = self.game.SignalMatrices  
      self.sstInvs = [  np.linalg.inv( s @ s.T ) for s in  self.SignalMatrices ] 
      self.Ps = [ self.SignalMatrices[i].T @  self.sstInvs[i] @ self.SignalMatrices[i] for i in range(self.N) ] 

      self.p0 =  [0.5, 0.5 ]
      self.sigma0 = np.identity(2)
      self.p = self.p0.copy()
      self.sigma = self.sigma0.copy()
      self.sigmaInv = np.linalg.inv( self.sigma0.copy() )

      self.sample_size = 100
      self.numC = 10
      self.samples = np.zeros( ( self.sample_size * self.numC * 2, self.M) )
      self.cInequality = self.getCells()
      self.ActiveActions = None
      self.n = np.zeros(self.N)

    def reset(self,):
      self.p0 =  [0.5, 0.5 ]
      self.sigma0 = np.identity(2)
      self.p = self.p0.copy()
      self.sigma = self.sigma0.copy()
      self.sigmaInv = np.linalg.inv( self.sigma0.copy() )
      self.samples = np.zeros( ( self.sample_size * self.numC * 2, self.M) )
      self.ActiveActions = None
      self.n = np.zeros(self.N)


    def update(self, action, feedback, outcome, X, t):
      # self.feedback_counter[action][outcome] +=1
      self.n[action] += 1

      curr_sigmaInv = self.sigmaInv
      self.sigmaInv = curr_sigmaInv + self.Ps[action]
      self.sigma = np.linalg.inv(  self.sigmaInv )

      Y_t = self.SignalMatrices[action] @ np.eye(self.M)[outcome]
      new_p = self.sigma @ ( curr_sigmaInv @ self.p + self.SignalMatrices[action].T  @ self.sstInvs[action] @ Y_t  )
      new_p = abs(new_p)
      self.p = new_p/sum(new_p)
      # print('probability', self.p, 'counter', self.n, 'sigma', self.sigma, 'means', self.p, 'new_p', new_p)

    def populateSamples(self, t, context = None):

      for i in range(self.sample_size):
        x = np.random.uniform(0, 1, self.M)
        stocX = x / x.sum()
        xnormMat = ( ( stocX - self.p ).T @ self.sigmaInv @ ( stocX - self.p) ) / np.log(t+2)
        xnorm = np.sqrt( xnormMat )
        # print('x', x, 'stocX', stocX, 'xnorm', xnorm)

        for j in range(self.numC):
          samp1 =  ( self.p + ( ( stocX  - self.p ) / xnorm ) * ( j / self.numC ) )
          samp2 = ( self.p - ( ( stocX - self.p ) / xnorm ) * ( j / self.numC ) ) 
          self.samples[ 2 * i * self.numC + 2 * j ] = samp1.T
          self.samples[ 2 * i * self.numC + 2 * j + 1 ] = samp2.T

    def getCells(self,):
      cInequality = [ ]
      for i in range(self.N):
        res = []
        for j in range(self.N):
          line = self.game.LossMatrix[i,...] - self.game.LossMatrix[j,...]
          res.append(line)
        res = np.vstack(res)
        cInequality.append( res )
      return cInequality

    def use_cache(self, t):
      return t%10 !=0 

    def get_action(self, t, context = None):

      if self.use_cache(t): #use previous p
        self.populateSamples(t)

      currentActiveActions = []
      score = np.zeros(self.N)
      for i in range(self.N):
          # print('samples', self.samples.T)
        temp = self.cInequality[i] @ self.samples.T
        boolMat =  temp <= np.zeros( (self.N, 2 * self.numC * self.sample_size) ) 
        sumBool = np.sum( boolMat, 0)
          # print('action', i, 'somme', sumBool)
        if np.max(sumBool) == self.N:
          currentActiveActions.append(i)
      score = { i: self.horizon - self.n[ i ] for i in currentActiveActions }
      # print('active actions', currentActiveActions, 'score', score,  )
      chosen = max(score, key=score.get)

      return chosen
