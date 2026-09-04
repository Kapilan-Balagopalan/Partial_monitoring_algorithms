
import numpy as np
from numpy.linalg import norm
# from torchvision import datasets, transforms
# import torch
import numpy as np
# from mnist_c import corruptions

class PolynomialContexts:
    def __init__(self, d, margin):
        self.d = d #number of features
        self.margin = margin #np.random.uniform(0,0.5) # decision boundary
        self.type = 'polynomial'
        self.d_context = 28

    def get_context(self, ):
        context = np.random.uniform(-1, 1,  self.d )
        return np.array(context).reshape(2,1)
    
    # while True:
    # x,y = context
    # if   y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 - self.margin  and label == 0:
    #     return np.array(context).reshape(2,1) #/ norm(context, 1)
    # elif   y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and label == 1:
    #     return np.array(context).reshape(2,1) #/ norm(context, 1)
    # def generate_unique_context(self,):
    #     self.context_A = []
    #     self.context_B = []
    #     while  len(self.context_A) == 0 or len(self.context_B) == 0:
    #         context = np.random.uniform(-1, 1, self.d)
    #         x,y = context
    #         if y < -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_A) == 0 :
    #             self.context_A = np.array(context).reshape(2,1) #/ norm(context, 1)
    #         elif y >= -0.25+x**2+0.1*x**3+0.05*x**4+0.01*x**5-0.4*x**6 + self.margin and len(self.context_B) == 0:
    #             self.context_B = np.array(context).reshape(2,1) #/ norm(context, 1)
    # def get_same_context(self, label):
    #     if label == 0:
    #             return self.context_A
    #     elif label == 1:
    #             return self.context_B

class LinearContexts:
    def __init__(self, w):
        self.d = len(w) #number of features
        self.w = w
        self.type = 'linear'
        self.normalize()

    def normalize(self,):
        all = []
        for _ in range(10000):
            c,d = self.get_context(False)
            all.append( c )
        all = np.array(all)
        self.mean = np.mean(all,0)
        self.std = np.std(all,0 )

    def get_context(self, normalize):
        context = np.random.uniform(0, 1,  self.d )
        context = np.array(context).reshape(self.d,1)
        val = self.w @ context
        distribution = [ val[0], 1-val[0] ]
        if normalize:
            context = ( context - self.mean ) / self.std
        return context, distribution

# class LinearContexts:
#     def __init__(self, w):
#         self.d = len(w) #number of features
#         self.w = w
#         self.type = 'linear'

#     def get_context(self, ):
#         context = np.random.uniform(0, 1,  self.d )
#         return np.array(context).reshape(self.d,1)
    
#     def get_distribution(self,cont):
#         val = self.w @ cont
#         return [ val[0], 1-val[0] ]
    
    # while True:
    # if   self.w.T @ context + self.b > self.margin and label == 0:
    #     return np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # elif  self.w.T @ context  + self.b < -self.margin and label == 1:
    #     return np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # def generate_unique_context(self,):
    #     self.context_A = []
    #     self.context_B = []
    #     while  len(self.context_A) == 0 or len(self.context_B) == 0:
    #         context = np.random.uniform(-1, 1, self.d)
    #         if self.w.T @ context + self.b > self.margin and len(self.context_A) == 0 :
    #             self.context_A = np.array(context).reshape(self.d,1) #/ norm(context, 1)
    #         elif self.w.T @ context + self.b < -self.margin and len(self.context_B) == 0:
    #             self.context_B = np.array(context).reshape(self.d,1) #/ norm(context, 1)
    # def get_same_context(self, label):
    #     if label == 0:
    #             return self.context_A
    #     elif label == 1:
    #             return self.context_B

class ToyContexts:

    def __init__(self, ):
        self.type = 'toy'
        self.d_context = 2

    def get_context(self, label):
        while True:
            context =   np.random.randint(2) # np.random.uniform(0, 1 )
            if   context >= 0.5 and label == 0:
                return np.array([1,context]).reshape(2,1) #/ norm(context, 1)
            elif  context < 0.5 and label == 1:
                return np.array([1,context-1]).reshape(2,1) #/ norm(context, 1)

    def generate_unique_context(self,):
        self.context_A = []
        self.context_B = []
        while  len(self.context_A) == 0 or len(self.context_B) == 0:
            context =  np.random.randint(2) # np.random.uniform(0, 1) #
            if context >= 0.5 and len(self.context_A) == 0 :
                self.context_A = np.array([1,context]).reshape(2,1) #/ norm(context, 1)
            elif context < 0.5 and len(self.context_B) == 0:
                self.context_B = np.array([1,context-1]).reshape(2,1) #/ norm(context, 1)

    def get_same_context(self, label):
        if label == 0:
                return self.context_A
        elif label == 1:
                return self.context_B
        

class OrthogonalContexts:

    def __init__(self, d):
        self.type = 'orthogonal'
        self.d = d

    def get_context(self, label):
        idx = np.random.randint(self.d)
        context = np.zeros( (self.d,1) )
        context[idx] = 1
        return context



# class MNISTcontexts():

#     def __init__(self, replacements, sampling_indexes):

#         self.horizon = len( sampling_indexes )
#         self.replacements = replacements 
#         self.sampling_indexes = sampling_indexes

#         self.switches = { '0':[6, 8, 9], '1':[4, 7], '2':[3], '3':[8, 9], '4':[8], '5':[6, 8], '6':[8], '7':[4, 8],'8':[0], '9':[8] }
        
#         # if digit_distribution == 'uniform':
#         self.digit_distribution =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
#         # elif digit_distribution == 'gaussian':
#         #     self.digit_distribution = [0.00938744, 0.02826442, 0.06660327, 0.12284535, 0.17736299, 0.2004605, 0.17736299, 0.12284535, 0.06660327, 0.02826442]
#         # else:
#         #     self.digit_distribution = [3.03931336e-02, 1.21241604e-01, 1.92058772e-01, 1.21241604e-01, 3.03931336e-02, 2.99782587e-03, 1.15116446e-04, 5.01558658e-01, 1.53276385e-07, 2.01265511e-11]
        
#     def get_contexts(self, data, outcomes):
#         contexts = np.empty( ( self.horizon, 784) )
#         # stream = np.empty( ( horizon, 785) )
#         labels = np.zeros( self.horizon)
#         outcomes = np.zeros( self.horizon)

#         for i, index, in enumerate( range( len(self.sampling_indexes) ) ) :

#             outcome = outcomes[i]
#             X, y =  data[index]
#             X = X.numpy()

#             if outcome == 1:
#                 #attacked_digit =  np.random.choice( [0,1,2,3,4,5,6,7,8,9], p= digit_distribution )
#                 candidates = self.switches[str(y)]
#                 replacement_digit = np.random.choice( candidates , p= np.ones( len(candidates)  ) / len(candidates) )
#                 choice = np.random.randint( 0, len(self.replacements[ replacement_digit ])-1 )
#                 replaced_image = self.replacements[replacement_digit][choice]
#                 X = replaced_image
#             contexts[i] = X.flatten()
#             labels[i] = y

#         return contexts
