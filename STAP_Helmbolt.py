import numpy as np
import pickle
from scipy.special import logit, expit
    


class STAP_Helmbolt():

    def __init__(self, game, d):

        self.name = 'helmbolt'
        self.d = d
        self.game = game
        self.N = game.n_actions

        self.nb_mistakes = 0
        self.lbd = 0.05


    def reset(self, ):

        self.nb_mistakes = 0

        self.contexts = []
        for i in range(self.N):
            self.contexts.append( {'features':[], 'labels':[], 'weights': None, 'V_it_inv': self.lbd * np.identity(self.d) } )

    def get_action(self, t, X,):

        
        if t>0:
            prediction = self.contexts[1]['weights'] @ X
            probability = expit(prediction)
            self.pred_action = 1 if probability < 0.5 else 0
            print(self.contexts[1]['weights'], X, self.contexts[1]['weights'].shape, X.shape)
            print('prediction', prediction, probability, self.pred_action)
        else:
            self.pred_action = 1

        r = np.random.uniform(0,1)
        flip_proba = np.sqrt( (1+self.nb_mistakes)/(t+1) )

        if self.pred_action == 0 and r <= flip_proba:
            action = 1
        else:
            action = self.pred_action
            
        return action

    def update(self, action, feedback, outcome, t, X):

        if action == 1:

            if outcome == self.pred_action:
                self.nb_mistakes += 1

            y_t = -1 if outcome==0 else 1

            self.contexts[action]['labels'].append( y_t )
            self.contexts[action]['features'].append( X )

            Y_it = np.array( self.contexts[action]['labels'] )
            X_it =  np.array( self.contexts[action]['features'] )
            

            Y_it =  Y_it.reshape(-1, 1).T
            X_it =  np.squeeze(X_it, 2).T
            print(Y_it.shape, X_it.shape)

            V_it_inv = self.contexts[action]['V_it_inv']
            self.contexts[action]['V_it_inv'] = V_it_inv - ( V_it_inv @ X @ X.T @ V_it_inv ) / ( 1 + X.T @ V_it_inv @ X ) 
            weights = Y_it @ X_it.T @ self.contexts[action]['V_it_inv']
            self.contexts[action]['weights'] = weights
        