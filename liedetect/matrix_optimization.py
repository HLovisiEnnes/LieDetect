import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from scipy.linalg import sqrtm

def project_sphere(data):
    n = np.shape(data)[1]
    np.random.shuffle(data)

    ys = []
    for i in range(len(data)):
        for j in range(i):
            ys.append((data[i],data[j]))
    
    X = cp.Variable((n,n), symmetric = True)

    constraints = [X >> 0]
    constraints += [cp.trace(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.sum([cp.abs(y[0].transpose() @ X @ y[0] - y[1].transpose() @ X @ y[1]) for y in ys])),
                      constraints)
    prob.solve()
    
    return X.value, sqrtm(X.value)