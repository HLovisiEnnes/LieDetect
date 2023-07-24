import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm

def project_sphere(data):
    X = np.cov(data.transpose(),bias=True) #notice that we take the transpose of the data to agree with the usual
    #ML notation of number_of_samples x ambient_dimension form for point cloud matrix representation
    
    return X, sqrtm(LA.pinv(X))