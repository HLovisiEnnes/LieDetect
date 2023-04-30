import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from scipy.linalg import sqrtm

def project_sphere(data):
    n = np.shape(data)[1]
    np.random.shuffle(data)
    
    X = cp.Variable((n,n), symmetric = True)

    constraints = [X >> 0]
    prob = cp.Problem(cp.Minimize(cp.sum([cp.abs(x.transpose() @ X @ x-1) for x in data])),
                      constraints)
    prob.solve()
    
    return X.value, sqrtm(X.value)