import cyipopt
import numpy as np
import scipy

class MLE_NLP:

    def __init__(self, game, A, feedback):
        self.game = game
        self.N = game.n_actions
        self.M = game.n_outcomes
        self.A = A
        self.feedbackMatrix = game.FeedbackMatrix_PMDMED
        self.feedback = feedback
        self.feedbackRowwise = [ sum(feedback[i]) for i in range(game.n_actions) ] # num of feedback a for action i
        # self.size = self.get_size()
        self.history = set([])
        self.counter = 0


  # Method to return the objective value */
    def objective(self, x):

      obj_value = 0
      for i in range(self.N): #calculate empirical divergence
        p_sym = np.zeros(self.A)
        hatp_sym = np.zeros(self.A)
        for j in range(self.M):
            p_sym[ self.feedbackMatrix[i][j] ] += x[j]
        for a in range(self.A):
            hatp_sym[a] = self.feedback[i][a] / self.feedbackRowwise[i] 
        
        kl_div = scipy.special.kl_div( hatp_sym , p_sym  ) 
        kl_div[np.isinf(kl_div)] = 0
        # print('hatp_sym', hatp_sym, 'p_sym', p_sym, 'kl div', kl_div )
        obj_value += self.feedbackRowwise[i] * kl_div.sum() 

        # if obj_value not in self.history:
        #     self.history.add(obj_value)
        # elif obj_value in self.history:
        #     self.counter+=1
        # if self.counter == 100:
        #     obj_value = None
        #     print('forced stop' )
        # print(obj_value)

      return obj_value


    def gradient(self, x):

        pMat = np.zeros( (self.N, self.A) )
        hatpMat = np.zeros( (self.N, self.A) )
        for i in range(self.N): #calculate empirical divergence
            for j in range(self.M):
                pMat[i][ self.feedbackMatrix[i][j] ] += x[j]
            for a in range(self.A):
                #print('self.feedback[i][a]', self.feedback[i][a],'self.feedbackRowwise[i]', self.feedbackRowwise[i] )
                hatpMat[i][a] = self.feedback[i][a] / self.feedbackRowwise[i] 
        #print(hatpMat)
        grad_f = np.zeros(self.M)
        for j in range(self.M):
            for i in range(self.N):
                a = self.feedbackMatrix[i][j] 
                grad_f[j] -= self.feedbackRowwise[i] * hatpMat[i][a] / pMat[i][a]
        # print('gradient', grad_f)
        return grad_f


    def constraints(self,x): # the sum of probability is equal to 1
        g = 0
        for j in range(self.M):
            g += x[j]
        return (g)

    # def get_size(self,):
    #     counter = 0
    #     for row in range(self.M):
    #         for col in range(row):
    #             counter+=1
    #     return counter

    def jacobian(self, x):
        values = np.zeros(self.M)
        for j in range(self.M):
                values[j] = 1
        #print('jacobian == run')
        return (values)

    def hessian(self, x, lagrange, obj_factor):
        # return the values. This is a symmetric matrix, fill the lower left triangle only
            
        pMat = np.zeros( (self.N, self.A) )
        hatpMat = np.zeros( (self.N, self.A) )
        for i in range(self.N): #calculate empirical divergence
            for j in range(self.M):
                pMat[i][ self.feedbackMatrix[i][j]   ] += x[j]
            for a in range(self.A):
                hatpMat[i][  a  ] = self.feedback[i][a] / self.feedbackRowwise[i] 

        # dense hessian
        values = np.zeros(self.size)
        idx = 0
        for row in range(self.M):
            for col in range(row):
                for i in range(self.N):
                    a1 = self.feedbackMatrix[i][row] 
                    a2 = self.feedbackMatrix[i][row]  
                    if a1 == a2 :

                        values[idx] += obj_factor * (self.feedbackRowwise[i] * hatpMat[i][a1] / (pMat[i][a1] * pMat[i][a1] ) )
                idx +=1

        #print('hessian == run')

        return values 

    # def intermediate( self, alg_mod, iter_count,  obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials ):
    #     # print("Objectivse value at iteration #%d is - %g" % (iter_count, obj_value) )
    #     pass

    # def feedback_idx(self, feedback):
    #     idx = None
    #     if self.N ==2:
    #         if feedback == 0:
    #             idx = 0
    #         elif feedback == 1:
    #             idx = 1
    #     elif self.N == 3:
    #         if feedback == 1:
    #             idx = 0
    #         elif feedback == 0.5:
    #             idx = 1
    #         elif feedback == 0.25:
    #             idx = 2
    #     else:
    #         if feedback == 1:
    #             idx = 0
    #         elif feedback == 2:
    #             idx = 1
    #     return idx


    



#   def eval_f(Index n, const Number* x, bool new_x, Number& obj_value):

#     obj_value = 0.;
#     for i in range(N): #calculate empirical divergence
#         p_sym = np.zeros(A)
#         hatp_sym = np.zeros(A)
#         for j in range(M):
#             p_sym[ feedbackMatrix[i,j] ] += x[j]
        
#         for a in range(A):
#             hatp_sym[a] = feedback[i, a] / feedbackRowwise[i]
        
#         # std::cout << "rw=" << feedbackRowwise(i) << " hatp=" << hatp_sym << " p=" << p_sym << std::endl;
#         obj_value += feedbackRowwise[i] * kl(hatp_sym, p_sym);
    
#     return True
    
  # Method to return the gradient of the objective 
#   def eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f): 

#     pMat = np.zeros(N,A)
#     hatpMat = np.zeros(N, A)
#     for i in range(N): #calculate empirical divergence
#         for j in range(M):
#             pMat[ i, feedbackMatrix[i, j] ] += x[j];
        
#         for a in range(A):
#             hatpMat[i, a)] = feedback[i, a] / feedbackRowwise[i]

#     for j in range(M):
#         grad_f[j] = 0.
#         for i in range(N):
#             a = feedbackMatrix[i, j]
#             grad_f[j] -= feedbackRowwise[i] * hatpMat[i, a] / pMat[i, a]


#     return True

  # Method to return:   1) The structure of the jacobian (if "values" is NULL)  2) The values of the jacobian (if "values" is not NULL)
#   def eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index *jCol, Number* values):

#     if values == None:

#         # return the structure of the jacobian
#         for j in range(M):
#             iRow[j] = 0
#             jCol[j] = j

#     else:

#         # return the values of the jacobian of the constraints
#         for j in range(M):
#             values[j] = 1;

#     return True

  # Method to return the constraint residuals */
#   def eval_g(Index n, const Number* x, bool new_x, Index m, Number* g):

#     g[0] = 0.
#     for j in range(M):
#         g[0] += x[j]

#     return True


#   # Method to return: 1) The structure of the hessian of the lagrangian (if "values" is NULL), 2) The values of the hessian of the lagrangian (if "values" is not NULL)
#   def eval_h(Index n, const Number* x, bool new_x,
#                       Number obj_factor, Index m, const Number* lambda,
#                       bool new_lambda, Index nele_hess, Index* iRow,
#                       Index* jCol, Number* values):

#     if values == None:
#         #return the structure. This is a symmetric matrix, fill the lower left triangle only.
#         # dense hessian
#         idx=0;
#         for row in range(M):
#         for col in range(row):
#             iRow[idx] = row
#             jCol[idx] = col
#             idx+=1
#     else:
#         # return the values. This is a symmetric matrix, fill the lower left triangle only
        
#         pMat = np.zeros(N, A)
#         hatpMat = np.zeros(N, A)
#         for i in range(N): #calculate empirical divergence
#         for j in range(M):
#             pMat[i, feedbackMatrix[i, j] ] += x[j];
        
#         for a in range(A):
#             hatpMat[i, a] = feedback[i, a] / feedbackRowwise[i]

#         # dense hessian
#         idx=0;
#         for row in range(M):
#         for col in range(row):
#             values[idx] = 0.
#             for i in range(N):
#                 a1 = feedbackMatrix(i, row)
#                 a2 = feedbackMatrix(i, row)
#                 if a1 == a2 :
#                     values[idx] += obj_factor * (feedbackRowwise(i) * hatpMat(i, a1) / (pMat(i, a1) * pMat(i,a1) ) );
 
#             #note: lambda[0] term is always zero 
#             idx += 1

#     return True


  # Solution Methods  This method is called when the algorithm is complete so the TNLP can store/write the solution */
#   def finalize_solution(SolverReturn status,
#                                  Index n, const Number* x, const Number* z_L, const Number* z_U,
#                                  Index m, const Number* g, const Number* lambda,
#                                  Number obj_value,
# 				                 const IpoptData* ip_data,
# 				                 IpoptCalculatedQuantities* ip_cq):

#   # here is where we would store the solution to variables, or write to a file, etc
#   # so we could use the solution.
#   # write result to global variable: we know this is not a good way...
#     ipopt_mle_solution = np.zeros(M)
#     for j in range(M):
#         ipopt_mle_solution[j] = x[j];

  

    # Name overloaded from TNLP, method to return some info about the nlp 
    # def get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style):
    # # number of variables
    # n = M
    # # number of constraints
    # m = 1
    # # number of non zeros in the jacobian
    # nnz_jac_g = M
    # # the hessian, total nonzeros, only need the lower left corner (since it is symmetric)
    # nnz_h_lag = (M*(M+1)) / 2;
    # # use the C style indexing (0-based)
    # index_style = TNLP::C_STYLE;
    # return True

#   # Method to return the bounds for my problem */
#   def get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u):
#     # the variables have lower bounds of 0.0
#     for j in range(M):
#         x_l[j] = 0.0001;
#     # the variables have upper bounds of 1.0
#     for j in range(M):
#       x_u[j] = 0.9999;
#     # the first constraint g1 has a lower bound of 25
#     g_l[0] = 25;
#     # the first constraint g1 has NO upper bound, here we set it to 2e19.
#     # Ipopt interprets any number greater than nlp_upper_bound_inf as
#     # infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
#     # is 1e19 and can be changed through ipopt options.
#     g_u[0] = 2e19;
#     # the second constraint g2 is an equality constraint, so we set the
#     # upper and lower bound to the same value
#     g_l[0] = g_u[0] = 1.0;
#     return True
  
