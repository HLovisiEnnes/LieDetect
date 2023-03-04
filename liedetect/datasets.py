import numpy as np
import scipy, sklearn
import velour
import matplotlib.pyplot as plt

' Functions to sample points '

def SampleOnTorus(dim = 2, ambient_dim = 6, n_points = 100, n_irreps = None,  freq = None, frequency_max = 2, conjugate = True, verbose = False, plot = False):
    '''
    Sample n point on an orbit of a representation of S_1^dim in R^ambient_dim.
    If dim>1, then by abelianity of the representation, the invariant subspaces must be equal
    
    Input:
        - dim: dimension of the torus
        - ambient_dim: dimension of the ambient space
        - n_points: number of points to sample
        - n_irreps: number of irreducible representations in the representation.
            If n_irreps==None, then it is chosen equal to int(ambient_dim/2)
        - freq: a list of lists with frequencies of normal decomposition
            If freq=None, the list is randomly constructed
        - frequencymax: maximal frequency of the representation
        - conjugate: if False, will be a representation in the canonical basis
        - verbose: whether to print comments
        - plot: if True, plots a 3D PCA scatter of data is shown
    Output:
        - Sample: a (n_points)x(ambient_dim) np matrix, the sampled points on the orbit
        - LieAlgebra: a list of length dim, containing (ambient_dim)x(ambient_dim) np matrices,
            the skew-symmetric matrices representing a derived Lie algebra basis
    Example:
        X, LieAlgebra = SampleOnTorus(dim=2, ambient_dim=4, n_points=5, n_irreps = 2, frequency_max=2, verbose=True)
        print(X)
        >>> [[ 1.16607538  0.47396429  0.59336947  0.25206892]
             [ 1.15010261 -0.04259541  0.81826323  0.07677827]
             [-0.51978319  0.60433863 -1.16810036 -0.01190787]
             [-1.19122581 -0.61834923 -0.34562944 -0.2813638 ]
             [-0.12375843  0.48702266 -1.30676828  0.19962326]]
        print(LieAlgebra[0])
        >>> [[ 2.72211718e-17  1.04590992e+00 -1.50345062e+00 -2.62213496e-01]
             [-1.04590992e+00 -3.87678742e-18 -7.00227388e-01  6.30826615e-01]
             [ 1.50345062e+00  7.00227388e-01 -4.49641091e-17  8.29874782e-01]
             [ 2.62213496e-01 -6.30826615e-01 -8.29874782e-01 -3.65981243e-18]]
        print(LieAlgebra[1])
        >>> [[-1.06880760e-18  8.29874782e-01 -6.30826615e-01 -7.00227388e-01]
             [-8.29874782e-01 -7.75357483e-18 -2.62213496e-01  1.50345062e+00]
             [ 6.30826615e-01  2.62213496e-01 -2.27076835e-17  1.04590992e+00]
             [ 7.00227388e-01 -1.50345062e+00 -1.04590992e+00 -3.80730947e-19]]    
    '''
    if n_irreps == None: n_irreps = int(ambient_dim/2)
    elif n_irreps>ambient_dim/2: raise ValueError('Ambient dimension is too little.')

    if verbose: print('-----> Representation of T^'+repr(dim), 'in R^'+repr(ambient_dim))
    
    # Define a canonical Lie algebra
    LieAlgebra = []
    if freq == None:
        for i in range(dim):
            # Define skew-symmetric matrix A
            A = np.zeros((ambient_dim,ambient_dim))
            frequencies = {j:random.randrange(1,frequency_max+1) for j in range(n_irreps)}
            for j in range(n_irreps): A[2*j,2*j+1], A[2*j+1,2*j] = frequencies[j], -frequencies[j]
            LieAlgebra.append(A)
            if verbose: print('Component', i+1, ' - Number of irreps in decomposition:', n_irreps, '- frequencies: ', list(frequencies.values()))
    
    else:
        if len(freq) != dim:
            raise Exception("ERROR: list of frequencies inconsistent with Lie Algebra dimension")
        else:
            for i in range(dim):
            # Define skew-symmetric matrix A
                A = np.zeros((ambient_dim,ambient_dim))
                frequencies = freq[i]
                for j in range(n_irreps): A[2*j,2*j+1], A[2*j+1,2*j] = frequencies[j], -frequencies[j]
                LieAlgebra.append(A)
                if verbose: print('Component', i+1, ' - Number of irreps in decomposition:', n_irreps, '- frequencies: ', freq[i])
            
    # Define origin Euclidean vector x
    x = np.zeros((ambient_dim,1))
    for i in range(n_irreps): x[2*i,0] = 1
        
    # Define orthogonal matrix O and conjugate
    if conjugate: O = scipy.stats.special_ortho_group.rvs(ambient_dim)
    else: O = np.eye(ambient_dim)
    for i in range(dim):
        LieAlgebra[i] = O @ LieAlgebra[i] @ O.T
        
    # Draw sample from uniform distribution
    Sample = []
    Angles = [np.random.uniform(0,2*np.pi,n_points) for i in range(dim)]
    for i in range(n_points):
        s = x.copy()
        for j in range(dim): s = scipy.linalg.expm(Angles[j][i]*LieAlgebra[j]) @ s
        Sample.append(s[:,0])
    Sample = np.array(Sample)
    
    if plot:
        velour.PlotPCA(Sample); plt.show();
    
    return Sample, LieAlgebra