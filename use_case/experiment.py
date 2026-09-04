import numpy as np
from multiprocessing import Pool
from functools import partial
# import plotly.graph_objects as go
import gzip
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.special import expit

import games
import randcbpside
import PGIDSratio
import utils
import os

def evaluate_parallel(nfolds, evaluator, alg, game):

    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=1))
    print('ncpus',ncpus)
    pool = Pool(processes=ncpus)
    partial_func = partial( evaluator.eval_policy_once, game, alg )
    return  pool.map( partial_func, range(nfolds) ) 



class Simulation():

    def __init__(self, M, imbalance):

        self.imbalance = imbalance
        self.M = M
        self.dim = len(imbalance)
        self.labels = [i for i in range(self.dim)]
        
    def get_contexts(self, horizon,):

        preds = []
        labels = []

        for i in range(horizon):

            label = np.random.choice( self.labels , p = self.imbalance)
            labels.append(label)
            pred_label = np.random.choice( self.labels , p = self.M[label] )
            preds.append(pred_label)
                
        return labels, preds


class Evaluation:

    def __init__(self , n_labels, stream, matrix, horizon):  
       self.n_labels = n_labels
       self.stream = stream
       self.matrix = matrix
       self.horizon = horizon

    def get_feedback(self, game, action, outcome):
        return game.FeedbackMatrix[ action ][ outcome ]

    def eval_policy_once(self, game, alg, jobid):

        np.random.seed(jobid)
        imbalance, M, ground_truth = utils.get_config(self.matrix, self.stream, self.n_labels)
        ground_truth = utils.get_ground_truth(M, imbalance)
        labels, preds = Simulation(M, imbalance ).get_contexts(self.horizon,)

        cum_regret = 0
        nb_verifications = 0

        alg.reset()
        t = 0
        
        error_counter = np.zeros( (10,10) )
        print('start the experiment')

        while (alg.greenlight == False).any(): 

            if t>=len(labels):
                break

            label, pred = labels[t], preds[t]
            
            context = np.zeros( (1, 10) )
            context[0][pred] = 1
            context = context.T

            outcome = 1 if pred == label else 0

            action = alg.get_action(t, context)
            # print(t, pred, action)
            
            if action != None:
                feedback =  self.get_feedback(game, action, outcome)
                alg.update(action, feedback, outcome, t, context)

            if action == 0:
                nb_verifications += 1 
                error_counter[label][pred] += 1
            else:
                nb_verifications += 0 

            i_star = np.argmin(  [ game.LossMatrix[i,...] @ np.array( [ ground_truth[label], 1-ground_truth[label] ] ) for i in [0,1] ]  )
            
            if action != None:
                loss_diff = game.LossMatrix[action,...] - game.LossMatrix[i_star,...]
                val = loss_diff @ np.array( [ ground_truth[label], 1 - ground_truth[label] ] )
                cum_regret += val
            else: 
                cum_regret += 0

            t+=1

        # print('finished', jobid, self.matrix, self.stream)

        if  alg.name == 'KFirst2' :
            params = alg.weights[:,1].tolist() 

        elif alg.name == 'cbpstrategy' or alg.name == 'randcbpstrategy':
            params = alg.weights
        
        return [ cum_regret, nb_verifications, params, error_counter, alg.decision, t ]


############# parrallel script:


from scipy.stats import norm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--algo", required=True, help="the algo to evaluate")
parser.add_argument("--horizon", required=True, help="the horizon")
parser.add_argument("--trials", required=True, help="the nb of trials")
parser.add_argument("--stream", required=True, help="the stream imbalance")
parser.add_argument("--matrix", required=True, help="the error imbalance")
args = parser.parse_args()


horizon = int( args.horizon )

proba = 0.01 #probability of the confidence intervals

algo = args.algo
stream = args.stream 
matrix = args.matrix 

n_trials = int( args.trials )
n_labels = 10

for threshold in [0.2, 0.1, 0.05, 0.025]: #detection threshold

    z_value = norm.ppf(1 - proba/2)
    margin = threshold / 10
    prior = 0.01
    nsample_wald = np.ceil( z_value**2 * ( prior * (1-prior) ) / margin**2 ) + 1 ## Wald confidence interval:

    alpha = 1.01 #cbp hyperparameter
    # nsample_cbp = np.ceil( np.exp( 1/(1-2*alpha) * np.log(proba/4) ) ) + 1

    # nsample = 2500

    if algo == 'kfirst2':
        game = games.label_efficient2( threshold )
        evaluator = Evaluation( n_labels, stream, matrix, horizon) 
        alg = utils.KFirst2(game, n_labels, nsample_wald, threshold)
        res = evaluate_parallel(n_trials, evaluator, alg, game)

    elif algo == 'randcbpstrategy':
        game = games.label_efficient2( threshold )
        evaluator = Evaluation( n_labels, stream, matrix, horizon) 
        alg = utils.CBPstrategy( game, True, n_labels, alpha, nsample_wald) 
        res = evaluate_parallel(n_trials, evaluator, alg, game)

    elif algo == 'cbpstrategy':
        game = games.label_efficient2( threshold )
        evaluator = Evaluation( n_labels, stream, matrix, horizon) 
        alg = utils.CBPstrategy( game, False, n_labels, alpha, nsample_wald) 
        res = evaluate_parallel(n_trials, evaluator, alg, game)

    else: 
        print('nothing')

    with gzip.open( './results/{}_{}_{}_{}_{}_{}.pkl.gz'.format(algo, horizon, n_trials, stream, matrix, threshold) ,'wb') as f:
        pkl.dump(res, f)









