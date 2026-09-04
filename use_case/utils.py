import numpy as np
from multiprocessing import Pool
from functools import partial
# import plotly.graph_objects as go
import gzip
import pickle as pkl

import matplotlib.pyplot as plt
import randcbp
import cbp

def confusion_matrix(M, errors, n_labels):

    for i in range(n_labels):

        M[i][i] -= errors[i]
        n_splits = np.random.randint(1, n_labels-1) 

        coefs = np.random.uniform(0, 1, n_splits)
        coefs = coefs / sum(coefs)

        for idx in range(n_splits):

            status = True
            while status == True:
                location = np.random.randint(0, n_labels)
                if location != i:
                    status = False

            M[i][location] += coefs[idx] *  errors[i]

    return M

def truncate(value):
    if value<-1 or value>1:
        status = False
        while status == False:
            value = abs( np.random.normal(0, 0.1 ) )
            if value <1:
                status = True
    return value

def get_ground_truth(M, imbalance): #proportion of error per predicted label
    # correct probabilities
    probas_correct = np.diag(M) * imbalance

    # correct probabilities
    probas_incorrect = M.copy()
    np.fill_diagonal( probas_incorrect, 0 ) 
    probas_incorrect = np.multiply(probas_incorrect, imbalance[:, np.newaxis])
    probas_incorrect = np.sum( probas_incorrect, 0)

    # final probabilites
    final_probas = probas_incorrect / (  probas_correct + probas_incorrect )
    final_probas = np.nan_to_num(final_probas, nan=0)
    return final_probas


def obtain_probability(t):

    epsilon = 10e-7# self.epsilon #10e-7
    M_prim = 8 #self.M_prim
    sigma = 1 #self.sigma
    alpha = 1.01

    U = np.sqrt( alpha  * np.log(t) ) 
    alphas = np.arange(0, U, U/M_prim )
    p_m_hat =  np.array([ np.exp( -(alphas[i]**2) / 2*(sigma**2)  )  for i in range(len(alphas)-1) ] )

    p_m = (1 - epsilon) * p_m_hat / p_m_hat.sum()
    p_m = p_m.tolist()
    p_m.append(epsilon)
        
    return alphas, p_m

def get_per_class2(mat, imbalance):
    matrix = mat.copy()
    # matrix = np.multiply(matrix, imbalance[:, np.newaxis])
    np.fill_diagonal(matrix, 0)
    res = np.sum( matrix , 1)
    return res

def get_per_class(mat):
    matrix = mat.copy()
    total = np.sum( matrix , 1)
    np.fill_diagonal(matrix, 0)
    mauvaise = np.sum( matrix , 1)
    res = mauvaise / total 
    res = np.nan_to_num(res, nan=0)
    return res

def get_config(matrix, stream, n_labels):

    if stream == 'balanced':
        # imbalance dans les donnees deployees:
        imbalance = np.ones(n_labels)#np.array( [ np.random.uniform(50, 75) if np.random.uniform(0,1)<0.1 else np.random.uniform(0,25) for _ in range(n_labels) ] )
        imbalance = imbalance / sum(imbalance)
    else:
        # imbalance dans les donnees deployees:
        imbalance = np.array( [ np.random.uniform(50, 75) if np.random.uniform(0,1)<0.1 else np.random.uniform(0,25) for _ in range(n_labels) ] )
        imbalance = imbalance / sum(imbalance)

    # if config == 'high' and matrix == 'non_uniform':
    #     per_class_error = [0.1]

    #     while sum(per_class_error) / len(per_class_error) < 0.1 or sum(per_class_error) / len(per_class_error) > 0.3 : 

    #         # confusion matrix:
    #         M = np.identity(n_labels)
    #         errors = np.array( [ truncate( abs( np.random.normal(0.5, 0.5 ) ) ) if np.random.uniform(0,1)<0.1 else truncate( abs( np.random.normal(0, 0.2 ) ) ) for _ in range(n_labels) ] )
    #         errors = np.round(errors,2)
    #         M = confusion_matrix(M, errors, n_labels)
                    
    #         # ground truth error probabilities:
    #         ground_truth = get_ground_truth(M, imbalance)
    #         per_class_error = get_per_class2(M, imbalance)
    #         # print('GT error rates', ground_truth)

    if matrix == 'nonuniform':
        ground_truth = [1]
        while sum(ground_truth) / len(ground_truth) > 0.1: 
            # confusion matrix:
            M = np.identity(n_labels)
            errors = np.array( [ truncate( abs( np.random.normal(0.1, 0.2 ) ) ) if np.random.uniform(0,1)<0.1 else truncate( abs( np.random.normal(0, 0.1 ) ) ) for _ in range(n_labels) ] )
            errors = np.round(errors,2)
            M = confusion_matrix(M, errors, n_labels)
                    
            # ground truth error probabilities:
            ground_truth = get_ground_truth(M, imbalance)
            # per_class_error = get_per_class2(M, imbalance)
            # print('GT error rates', ground_truth)

    elif matrix == 'uniform':
        error = np.random.uniform(0,0.1)
        errors = np.ones(n_labels) * error
        M = np.identity(n_labels)
        for i in range(n_labels):
            for j in range(n_labels):
                if i == j:
                    M[i][j] -= errors[i]
                else:
                    M[i][j] = errors[i] / (n_labels-1)
        ground_truth = get_ground_truth(M, imbalance)

    elif matrix == 'adversarialattack':
        error = np.random.uniform(0,0.1)
        errors = np.ones(n_labels) * error
        errors[-1] = 0
        M = np.identity(n_labels)
        for i in range(n_labels):
            for j in range(n_labels):
                if i == j:
                    M[i][j] -= errors[i]
                elif j == n_labels-1:
                    M[i][j] = errors[i] 
        ground_truth = get_ground_truth(M, imbalance)

    elif matrix == 'slidingerror':
        error = np.random.uniform(0,0.1)
        errors = np.ones(n_labels) * error
        errors[-1] = 0
        M = np.identity(n_labels)
        for i in range(n_labels):
            for j in range(n_labels):
                if i == j and j<n_labels-1:
                    M[i][j] -= errors[i]
                    M[i][j+1] = errors[i] 
        ground_truth = get_ground_truth(M, imbalance)


    return imbalance, M, ground_truth



class CBPstrategy(): ### C-CBP and C-RandCBP

    def __init__(self, game, rand, n_labels, alpha, nsamples):
        self.nsamples = nsamples

        if rand:
            self.name = 'randcbpstrategy'
        else:
            self.name = 'cbpstrategy'

        self.n_labels = n_labels
        self.game = game
        self.alpha = alpha
        self.rand = rand

        if rand:
            self.CBPs = [ randcbp.RandCBP(self.game, self.alpha, 1, 10, 10e-7) for _ in range(self.n_labels) ]
        else:
            self.CBPs = [ cbp.CBP(game, self.alpha ) for _ in range(self.n_labels) ]

        self.weights = [  0  for idx in range(n_labels) ]
        self.greenlight = np.array( [False] * self.n_labels )
        self.decision = [None] * self.n_labels
        self.nb_calls = np.zeros(self.n_labels)
        self.nb_verifs = np.zeros(self.n_labels)
    
    def get_action(self, t, context  ):
        # print('nb calls', t, np.argmax(context), self.nb_calls, self.greenlight)
        idx = np.argmax(context)

        if self.nb_calls[idx] == 0:
            # print('step1')
            action = 0 

        elif self.nb_calls[idx] == 1:
            # print('step2')
            action = 1

        elif self.nb_calls[idx]>1 and self.greenlight[idx] == False: 
            # print('step3')
            action = self.CBPs[idx].get_action( self.nb_calls[idx] )
            self.CBPs[idx].get_decision( self.nb_calls[idx] )
            self.decision[idx] = self.CBPs[idx].decision[0] if len(self.CBPs[idx].decision) == 1 else self.decision[idx]
            
        elif self.nb_calls[idx]>1 and self.greenlight[idx] == True:
            # print('step4')
            action = None

        return action

    def update(self, action, feedback, outcome, t, context):
        
        idx = np.argmax(context)
        self.CBPs[idx].update(action, feedback, outcome, context, self.nb_calls[idx]  )
        if self.nb_verifs[idx] > 1:
            val = self.CBPs[idx].nu[0]  / self.CBPs[idx].n[0] 
            self.weights[idx] = val[1][0]
            
        self.nb_calls[idx] += 1
        if action == 0:
            self.nb_verifs[idx] += 1

        if (self.nb_calls[idx]>self.nsamples and self.decision[idx] != None): # or self.nb_verifs[idx]>self.max_verif
            self.greenlight[idx] = True 

    def reset(self,):
        if self.rand:
            self.CBPs = [ randcbp.RandCBP(self.game, self.alpha, 1, 10, 10e-7) for _ in range(self.n_labels) ]
        else:
            self.CBPs = [ cbp.CBP(self.game, self.alpha ) for _ in range(self.n_labels) ]

        self.weights = [  0  for idx in range(self.n_labels) ]
        self.greenlight = np.array( [False] * self.n_labels )
        self.decision = [None] * self.n_labels
        self.nb_verifs = np.zeros(self.n_labels)
        self.nb_calls = np.zeros(self.n_labels)


class KFirst2(): #### Explore-fully

    def __init__(self, game, n_labels, nsamples, threshold):
        
        self.name = 'KFirst2'
        self.n_labels = n_labels
        self.game = game
        self.nsamples = nsamples
        self.threshold = threshold
        # self.max_verif = 10000 / self.n_labels
        self.weights = np.ones( (self.n_labels,2) ) * 1/2
        self.feedbacks = np.zeros( (self.n_labels,2) )
        self.N = np.zeros( self.n_labels )
        self.greenlight = np.array( [False] * self.n_labels )
        self.decision = [None] * self.n_labels
        self.nb_calls = np.zeros(self.n_labels)
        self.nb_verifs = np.zeros(self.n_labels)

    def get_action(self, t, context = None ):
        idx = np.argmax(context)
        action = None if self.greenlight[idx] == True else 0
        return action

    def update(self, action, feedback, outcome, t, context):
        
        idx = np.argmax(context)
        self.feedbacks[idx][feedback] += 1
        self.N[idx] += 1
        estimates = [ self.feedbacks[i] / self.N[i] if self.N[i] !=0 else np.ones( (1,2) )*1/2 for i in range(len(self.N) ) ] 
        self.weights = np.vstack(estimates)

        self.decision[idx] = 0 if self.weights[idx][1] > self.threshold else 1


        self.nb_calls[idx] += 1
        if action == 0:
            self.nb_verifs[idx] += 1

        if (self.nb_calls[idx]>self.nsamples and self.decision[idx] != None): # or self.nb_verifs[idx]>self.max_verif
            self.greenlight[idx] = True 

    def reset(self,):
        self.weights = np.ones( (self.n_labels,2) )
        self.feedbacks = np.zeros( (self.n_labels,2) )
        self.N = np.zeros( self.n_labels )
        self.greenlight = np.array( [False] * self.n_labels )
        self.decision = [None] * self.n_labels
        self.nb_verifs = np.zeros(self.n_labels)
        self.nb_calls = np.zeros(self.n_labels)

