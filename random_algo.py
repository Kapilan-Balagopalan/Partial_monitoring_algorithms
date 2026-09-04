

import numpy as np


class Random():

    def __init__(self, game, horizon):

        self.game = game
        self.horizon = horizon
        self.N = game.n_actions

    def get_action(self, t, context = None ):
        
        pbt = np.ones( self.game.n_actions ) / self.game.n_actions
        action = np.random.choice(self.game.n_actions, 1,  p = pbt )[0]
        return action

    def update(self, action, feedback, outcome, context, t):
        pass

    def reset(self,):
        pass
