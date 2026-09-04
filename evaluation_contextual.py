from sklearn.preprocessing import PolynomialFeatures
import numpy as np


class Evaluation_contextual:

    def __init__(self, horizon ):

        self.horizon = horizon

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, alg, game, job):#jobid
        context_generator, seed = job
        np.random.seed(seed)
        
    
        cumRegret =  np.zeros(self.horizon, dtype =float)
        actions = np.zeros(self.horizon, dtype =float)


        for t in range(self.horizon):
            print(t)
            context, distribution = context_generator.get_context(True)
            print(distribution)
            # distribution = context_generator.get_distribution(context)
            # outcome = np.random.choice( 2 , p = distribution )
            outcome = 0 if distribution[0]<0.5 else 1
            distribution = np.array([1-outcome,outcome])

            action = alg.get_action(t, context)
            
            feedback =  self.get_feedback( game, action, outcome )

            alg.update(action, feedback, outcome, t, context )
            print('t', t, 'action', action, 'outcome', outcome,  )

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( distribution ) for i in range(alg.N) ]  )
            loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
            val = loss_diff @ np.array( distribution )
            
            print(action, outcome, val)

            cumRegret[t] =  val
            actions[t] = action

        return  np.cumsum( cumRegret ) 