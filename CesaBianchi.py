import numpy as np
import pickle
from scipy.special import logit, expit
    


class CesaBianchi():

    def __init__(self, game, d):

        self.name = 'helmbolt'
        self.d = d
        self.game = game
        self.N = game.n_actions

        self.K = 0
        self.beta = 1
        self.lbd = 0.05


    def reset(self, ):

        self.K = 0
        self.norm_hist = 0
        self.contexts = {'features':[], 'labels':[], 'weights': None, 'V_it_inv': self.lbd * np.identity(self.d) } 

    def get_action(self, t, X,):

        norm = np.linalg.norm( X )
        self.X_prime = max( self.norm_hist, norm  )

        if t>0:
            # print(X.shape,  self.contexts['weights'])
            prediction = self.contexts['weights'] @ X
            probability = expit(prediction)
            self.pred_action = 1 if probability < 0.5 else 2
            # print(self.contexts['weights'], X, self.contexts['weights'].shape, X.shape)
            print('prediction', prediction, 'proba',probability, 'pred action', self.pred_action)
        else:
            self.pred_action = 0
            probability = 1

        b = self.beta * np.sqrt(self.K+1) * self.X_prime**2
        # b = self.beta * np.sqrt(1+self.K) 
        p = b / ( b + abs( probability ) )

        self.Z = np.random.binomial(1, p)
        self.Z = 1-self.Z

        if self.Z == 1:
            action = 0
        else:
            action = self.pred_action

        return action

    def update(self, action, feedback, outcome, t, X):

        if action==0:

            if (self.pred_action == 1 and outcome == 0) or (self.pred_action == 2 and outcome ==1):
                self.K += 1
                self.norm_hist = self.X_prime**2

            y_t = 1 if outcome==0 else -1

            self.contexts['labels'].append( y_t )
            self.contexts['features'].append( X )

            Y_it = np.array( self.contexts['labels'] )
            X_it =  np.array( self.contexts['features'] )
            

            Y_it =  Y_it.reshape(-1, 1).T
            X_it =  np.squeeze(X_it, 2).T
            print(Y_it.shape, X_it.shape)

            V_it_inv = self.contexts['V_it_inv']
            self.contexts['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 
            weights = Y_it @ X_it.T @ self.contexts['V_it_inv']
            self.contexts['weights'] = weights
        