import numpy as np
from polyagamma import random_polyagamma
from scipy.special import expit
import time
import signal 

# Define the handler function to stop the computation
def timeout_handler(signum, frame):
    raise TimeoutError("Computation timed out")

class PGIDSratio():
    def __init__(self, game, d):

        self.game = game
        self.N = game.n_actions

        self.gibbsits = 10
        self.d = d
        self.pmean = np.ones(self.d) * 0.5
    
        self.pcovar = np.identity( self.d )
        self.pcovar_inv = np.linalg.inv(self.pcovar)

        self.contexts = {'features':[], 'labels':[] } 
        self.current_thetamat = np.zeros( ( self.gibbsits, self.d ) )

    def thetagibbs(self, contexts, outcomes,t):


        thetamat = np.zeros( ( self.gibbsits, self.d ) )
        thetamat[0] = self.initial_sample
        kappa = np.array(outcomes) - 0.5
        features = np.array(contexts)
        features = np.squeeze(features, 2)

        for m in range(1,self.gibbsits):
            comp =  features @ thetamat[m-1]
            omega = random_polyagamma( 1 , comp )
            Omegamat = np.diag( omega ) 
            matrix = features.T @ Omegamat @ features + self.pcovar
            Vomega   = np.linalg.inv( matrix ) #variance
            momega   = Vomega @ ( features.T @ kappa + self.pcovar_inv @ self.pmean ) #mean
            thetamat[m] = np.random.multivariate_normal(momega, Vomega, 1)  #np.array([[0,1]]) # 

        return thetamat

    def rewfunc(self, action, param, context):
        # print('estimation', param.T @ context, 'sampled param', param, 'context', context, 'logit', expit( param.T @ context )  )
        # print('sampled param', param.shape, 'context', context.shape)
        # print( 'p1', expit( param @ context ),'p2', expit( param @ n_context ) )
        if action == 0:
            res = self.game.LossMatrix[0,0] * expit( param @ context ) + self.game.LossMatrix[0,1] * (1 - expit( param @ context ) )
        else:
            res = self.game.LossMatrix[1,0] * expit( param @ context ) + self.game.LossMatrix[1,1] * (1 - expit( param @ context ) )
        return res

    def get_action(self,  t, X):

        if t == 0:
            # Always intervene on the first anomaly to get some data
            action = 1
            self.initial_sample = self.pmean

        else:
            # Gibbs sampling, Start the timer
            allowed_time = 10 #(in seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(allowed_time)

            try:
                self.thetasamples = self.thetagibbs( self.contexts['features'], self.contexts['labels'],t )
                self.initial_sample = self.thetasamples[-1]
                # print('samples', self.thetasamples)
                # Compute gap estimates
                delta0 = np.zeros(self.gibbsits)
                delta1 = np.zeros(self.gibbsits)
                for j in range(self.gibbsits):
                    action0 = self.rewfunc( 0, self.thetasamples[j], X)
                    action1 = self.rewfunc( 1, self.thetasamples[j], X)
                    minimum = min( action0, action1 )
                    delta0[j] =  action0 - minimum
                    delta1[j] =  action1 - minimum
                    # print('action0',action0,'action1', action1,'minimum', minimum)
                    
                # print('delta0s',delta0)
                # print('delta1s',delta1)
                deltaone = np.mean(delta1)
                deltazero = np.mean(delta0)
                # print('delta0', deltazero, 'delta1', deltaone)
                #Compute expected information gain
                tuneidsparam2 = min(1, deltazero / ( abs(deltaone-deltazero) ) ) 
                # print(tuneidsparam2)
                p= max(0, tuneidsparam2 )
                print(p)
                action = np.random.choice( [0, 1], p=[1-p, p] )
                # print('t',t, 'action', action, 'proba', max(0, min(1,tuneidsparam2) ), 'idsparam', tuneidsparam2, 'delta0', deltazero / ( abs(deltazero-deltaone) ) )
            except TimeoutError:
                # Handle the timeout error
                print('hey')
                action = None 

        return action

    def update(self,action, feedback, outcome, t, context):

        if action ==1 :
            self.contexts['features'].append( context )
            self.contexts['labels'].append( 1-outcome )

    def reset(self,):
        self.n = np.zeros( self.N )
        self.contexts = {'features':[], 'labels':[] } 
        self.current_thetamat = np.zeros( ( self.gibbsits, self.d ) )

