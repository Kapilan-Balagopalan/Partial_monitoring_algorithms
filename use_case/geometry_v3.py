import numpy as np
import gurobipy as gp
from gurobipy import GRB
import collections

def alphabet_size(FeedbackMatrix, N,M):
    alphabet = []
    for i in range(N):
        for j in range(M):
            alphabet.append(FeedbackMatrix[i][j])
    return len(set(alphabet)) 

def calculate_signal_matrices(FeedbackMatrix, N,M,A):
    signal_matrices = []
    for i in range(N):
        signalMatrix = np.zeros( (A,M) )
        for j in range(M):
            a = FeedbackMatrix[i][j]
            signalMatrix[a][j] = 1
        signal_matrices.append(signalMatrix)
    return signal_matrices

def getParetoOptimalActions(LossMatrix, N, M, halfspace):
    actions = []
    for z in range(N):
        feasible = True

        m = gp.Model( )
        m.Params.LogToConsole = 0

        vars = []
        for i in range(M):
            varName =  'p_{}'.format(i) 
            vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName) )
            m.update()

        simplexExpr = gp.LinExpr()
        for j in range(M):
            simplexExpr += 1.0 * vars[j]

        m.addConstr(simplexExpr == 1.0, "css")

        for i2 in range(N):
            if(i2 != z):
                lossExpr = gp.LinExpr()
                for j in range(M):
                    lossExpr += ( LossMatrix[i2][j] - LossMatrix[z][j] ) * vars[j]
                    lossConstStr = 'c {}'.format(i2)
                m.addConstr(lossExpr >= 0.0, lossConstStr )

        for element in halfspace:
            pair, sign = element[0], element[1]
            if sign != 0:
                halfspaceExpr = gp.LinExpr()
                for j in range(M):
                    coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
                    if coef != 0:
                        halfspaceExpr += coef * vars[j]
                halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
                m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
        try:
            m.optimize()
            objval = m.objVal
        except:
            feasible=False

        if feasible:
            actions.append(z)

    return actions

def getNeighborhoodActions(LossMatrix, N, M, halfspace,mathcal_N):
    actions = []
    for pair in mathcal_N:
        i1,i2 = pair
        if isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
            actions.append( [i1,i2] )
    return actions

def isNeighbor(LossMatrix, N, M, i1, i2, halfspace):
    feasible = True


    m = gp.Model( )
    m.Params.LogToConsole = 0
    vars = []
    for j in range(M):
        varName = "p {}".format(j)
        vars.append( m.addVar(0.00001, 1.0, -1.0, GRB.CONTINUOUS, varName ) )
        m.update()

    simplexExpr = gp.LinExpr()
    for j in range(M):
        simplexExpr += 1.0 * vars[j]
    m.addConstr(simplexExpr == 1.0, "css") 

    twoDegenerateExpr = gp.LinExpr()
    for j in range(M):
        twoDegenerateExpr += (LossMatrix[i2][j]-LossMatrix[i1][j]) * vars[j]
    m.addConstr(twoDegenerateExpr == 0.0, "cdeg")

    for i3 in range(N):
        if( (i3 == i1) or (i2 == i1) ):
            pass
        else:
            lossExpr = gp.LinExpr()
            for j in range(M):
                lossExpr += ( LossMatrix[i3][j]-LossMatrix[i1][j] ) * vars[j]
            lossConstStr = "c".format(i3)
            m.addConstr(lossExpr >= 0.0, lossConstStr )

    for element in halfspace:
        pair, sign = element[0], element[1]
        if sign != 0:
            halfspaceExpr = gp.LinExpr()
            for j in range(M):
                coef = sign * (LossMatrix[ pair[0] ][j]-LossMatrix[  pair[1] ][j] ) 
                if coef != 0:
                    halfspaceExpr += coef * vars[j]
            halfspaceConstStr = "ch_{}_{}".format( pair[0] ,pair[1] )
            m.addConstr(halfspaceExpr >= 0.001, halfspaceConstStr )
    try:
        m.optimize()
        objval = m.objVal
    except:
        feasible = False

    return feasible


def getV(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, mathcal_N, V):
    v = collections.defaultdict(dict)
    for pair in mathcal_N:
        v[ pair[0] ][ pair[1] ]  = getVij(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, V,  pair[0], pair[1])
    return v
  
def getVij(LossMatrix, N, M, FeedbackMatrix, SignalMatrices, V, i1, i2):

    l1 = LossMatrix[i1]
    l2 = LossMatrix[i2]
    ldiff = l1 - l2

    m = gp.Model( )
    m.Params.LogToConsole = 0

    vars = collections.defaultdict(dict)
    for k in V[i1][i2] :
        vars[k] = []
        sk = len( set(FeedbackMatrix[k]) )
        for a in range( sk ):
            varName = "v_{}_{}_{}".format(i1, i2, a) 
            vars[k].append( m.addVar(-GRB.INFINITY, GRB.INFINITY, 0., GRB.CONTINUOUS, varName ) ) 
            m.update()

    obj = 0
    for k in  V[i1][i2] :
        sk = len( set(FeedbackMatrix[k]) )
        for a in range( sk ):
            obj += vars[k][a]**2
    m.setObjective( obj , GRB.MINIMIZE)

    expression = 0
    for k in  V[i1][i2] :
        expression += SignalMatrices[k].T @ vars[k]
    for l in range(len(ldiff)):
        m.addConstr( expression[l] == ldiff[l],  'constraint{}'.format(l) )

    m.optimize()
    
    vij = {}
    for k in V[i1][i2] :
        sk = len( set(FeedbackMatrix[k]) )
        vijk = np.zeros( sk )
        for a in range( sk ):
            vijk[a] =   vars[k][a].X
        vij[k] = vijk

    return vij


def getConfidenceWidth( mathcal_N, V, v,  N):
    W = np.zeros(N)

    for pair in mathcal_N:
        for k in V[ pair[0] ][ pair[1] ]:
            vec = v[ pair[0] ][ pair[1] ][k]
            W[k] = np.max( [ W[k], np.linalg.norm(vec, np.inf) ] )
    return W

  
def f(t, alpha):
    return   (t**(2/3) ) * ( alpha * np.log(t) )**(1/3)

  
        
        

