import geometry_v3
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class TSPM_alg:

    def __init__(self, game, horizon, R ):
      self.game = game
      self.horizon = horizon
      self.R = R

      self.N = game.n_actions
      self.M = game.n_outcomes
      self.A = geometry_v3.alphabet_size(game.FeedbackMatrix_PMDMED, self.N, self.M)
      print('n-actions', self.N, 'n-outcomes', self.M )

      self.SignalMatrices = game.SignalMatricesAdim
      self.sts = [  s.T @ s  for s in  self.SignalMatrices ] 

      self.lbd = 0.001
      self.B0 = self.lbd * np.identity(self.M)
      self.b0 = np.zeros( (self.M, 1) ) 
      self.B = self.B0
      self.b = self.b0

      self.n , self.q = [] , []
      for _ in range(self.N):
          self.n.append( np.zeros( self.A ) )
          self.q.append(  np.zeros( self.A ) )

    def in_simplex(self,p):
        res = False
        if p.sum()<=1 and (p<=1).all() and (p>=0).all():
            res = True
        return res

    def truncated_multivariate_gaussian_density(self, x):
        lower = np.zeros(self.M)
        upper = np.ones(self.M)
        # print(self.b.squeeze(), self.B)
        mvn = multivariate_normal( mean=np.zeros(self.M), cov=self.lbd * np.identity(self.M) ) #(mean=self.b.squeeze(), cov=self.B )
        # lower_cdf = mvn.cdf(lower)
        # upper_cdf = mvn.cdf(upper)
        # normalization_constant = upper_cdf - lower_cdf
        density = mvn.pdf(x) #/ normalization_constant
        # Return the density value
        return density

    def reset(self,):
        self.B0 = self.lbd * np.identity(self.M)
        self.b0 = np.zeros( (self.M, 1) ) 
        self.B = self.B0
        self.b = self.b0
        self.n , self.q = [] , []
        for _ in range(self.N):
            self.n.append( np.zeros( self.A ) )
            self.q.append(  np.zeros( self.A ) )

    def sample_from_g(self,):

        condition = False

        one_vec = np.atleast_2d( np.ones(self.M - 1) ).T 
        C = self.B[:self.M-1, :self.M-1]
        d =  np.atleast_2d(self.B[:self.M-1, -1]).T 

        D = (np.dot(d, one_vec.T) + np.dot(one_vec, d.T)) / 2
        f = self.B[-1, -1]

        b_alpha = np.atleast_2d(self.b[:self.M-1,0 ]).T 
        b_M = self.b[-1, 0] 

        B_tilde = C - 2 * D + f * np.dot(one_vec, one_vec.T)
        b_tilde = f * one_vec - d + b_alpha - b_M * one_vec

        B_tilde_inv = np.linalg.inv(B_tilde)
        Bb = B_tilde_inv @ b_tilde

        while condition == False: 
            p =  np.random.multivariate_normal(  Bb[:,0],  B_tilde_inv  )
            if self.in_simplex(p):
                condition = True
        
        return  np.concatenate( [ p , [1 - p.sum()] ] )

    def accept_reject(self,t):
        limit = 100
        threshold = 0
        while threshold < limit:
            p_tilde = self.sample_from_g()
            u_tilde = np.random.uniform(0, 1)
            threshold += 1
            F = self.F( p_tilde )
            G = self.G( p_tilde )
            # print('F', F, 'G', G, 'F/G', F/G)
            if self.R * u_tilde <  F / G :
                #print('rejects', threshold)
                return p_tilde
            if threshold == limit-1:
                print('limit threshold ', t)
                return [0.5] * self.M

    def F(self, p):
        # mean_vec = np.linalg.inv( self.B ) @ self.b
        # inv = np.linalg.inv( self.B )
        # result = multivariate_normal(mean=  mean_vec[:,0], cov= inv ).pdf( p ) 
        result = self.truncated_multivariate_gaussian_density(p)
        # print('density', result)

        for i in range(self.N):
            result *= np.exp( - self.n[i].sum() * scipy.special.kl_div( self.q[i] , self.SignalMatrices[i] @ p  ).sum() )

        return result

    def G(self,p):
        
        # result = self.truncated_multivariate_gaussian_density(p)
        # mean_vec = np.linalg.inv( self.B ) @ self.b
        # inv = np.linalg.inv( self.B )
        # result = multivariate_normal(mean=  mean_vec[:,0], cov= inv ).pdf( p ) 
        result = self.truncated_multivariate_gaussian_density(p)
        

        for i in range(self.N):
            result *= np.exp( -1/2 * self.n[i].sum() * np.linalg.norm( self.q[i]  - self.SignalMatrices[i] @ p ) ** 2  )
        
        return result

    def get_action(self, t, context = None):
        
        if t < self.N :
            action = t
        else:
            p_tilde = self.accept_reject(t)
            # print(p_tilde)
            action = np.argmin(  self.game.LossMatrix @ p_tilde  ) 
        # print('action', action, p_tilde)

        return action

    def update(self,  action, feedback, outcome, X, t):

      feedback = self.game.FeedbackMatrix_PMDMED[action][outcome]

      self.B = self.B + self.sts[action]
       
      e_y = np.zeros( (self.A, 1) )
      e_y[ feedback ] = 1

    #   value = self.SignalMatrices[action].T @ e_y
    #   print('value', value, value.shape, self.b.shape )
      self.b = self.b + self.SignalMatrices[action].T @ e_y 

      self.n[action][ feedback ] += 1
      self.q[action] = self.n[action] / self.n[action].sum() 