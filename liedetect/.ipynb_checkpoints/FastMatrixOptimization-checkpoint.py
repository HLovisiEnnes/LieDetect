import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm

def project_sphere(data):
    X = np.cov(data,bias=True)
    
    return X.value, sqrtm(LA.pinv(X))