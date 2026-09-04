
import numpy as np
import geometry 

class FeedExp3():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon
        self.pbt = np.ones(self.game.n_actions)/self.game.n_actions

        self.u = np.ones(self.game.n_actions)/self.game.n_actions

        self.eta, self.gamma = self.parameters_Piccolboni()

    def reset(self,):
        self.eta, self.gamma = self.parameters_Piccolboni()
        self.u = np.ones(self.game.n_actions)/self.game.n_actions
        self.pbt = np.ones(self.game.n_actions)/self.game.n_actions

        
        
    def get_action(self, t):

        self.pbt_hat =  (1 - self.gamma) * self.pbt  + self.gamma * self.u 
        # print('pbt', self.pbt_hat, 'gamma', self.gamma)

        action = np.random.choice(self.game.n_actions, 1,  p = self.pbt_hat )[0]
        return action

    def update(self,  action, feedback, outcome, X, t):

        x = np.array( [ feedback * self.game.LinkMatrix[i][action] / self.pbt_hat[i] for i in range(self.game.n_actions) ] )
        Z = sum( self.pbt / np.exp( self.eta * x ) )
        self.pbt = self.pbt / ( Z * np.exp( self.eta * x  ) )

    def parameters_Piccolboni(self, ):
        ## [Piccolboni Schindelhauer "Discrete Prediction Games with Arbitrary Feedback and Loss" 2000]
        ## fixed-known-horizon setting
        eta = pow( np.log(self.game.n_actions), 1./2.) / pow(self.horizon, 1./2.)
        gamma = np.fmin(1., pow(self.game.n_actions, 1./2.) * pow( np.log(self.game.n_actions),1./4.) / pow(self.horizon, 1./4.))
        return eta, gamma
