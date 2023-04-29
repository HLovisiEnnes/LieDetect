import autograd.numpy as np 
import scipy, sklearn
import velour
import matplotlib.pyplot as plt
import random
from .matrix_manipulation import *
' Functions to sample points '

#sample on torus, both sample knots and torus T^n orbits
' Functions to sample points '

def SampleOnTorus(dim=2, ambient_dim=6, n_points=100, frequencies=None, frequency_max=2, conjugate=True, right_multiply=True, verbose=False):
    '''
    Sample n point on an orbit of a representation of S_1^dim in R^ambient_dim.
    If dim>1, then by abelianity of the representation, the invariant subspaces must be equal
    
    Input:
        - dim: dimension of the torus
        - ambient_dim: dimension of the ambient space
        - n_points: number of points to sample
        - n_irreps: number of irreducible representations in the representation.
            If n_irreps==None, then it is chosen equal to int(ambient_dim/2)
        - frequencymax: maximal frequency of the representation
        - conjugate: if False, will be a representation in the canonical basis
        - verbose: whether to print comments
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
    if verbose: print('-----> Representation of T^'+repr(dim), 'in R^'+repr(ambient_dim))
    
    # Define frequencies
    if frequencies==None: frequencies = random.choice(GetFrequenciesToTest(dim, ambient_dim, frequency_max))
            
    # Define a canonical Lie algebra
    LieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)   
    print('frequencies:', frequencies)
        
    # Make basis orthonormal    
    LieAlgebra = GramSchmidtOrthonormalizationMatrices(LieAlgebra)
        
    # Define orthogonal matrix P and right-multiply
    if dim>1 and right_multiply:
        P = scipy.stats.special_ortho_group.rvs(dim)
        LieAlgebra = [np.sum([LieAlgebra[j]*P[j,i] for j in range(dim)],0) for i in range(dim)]
            
    # Define orthogonal matrix O and conjugate
    if conjugate: O = scipy.stats.special_ortho_group.rvs(ambient_dim)
    else: O = np.eye(ambient_dim)
    for i in range(dim): LieAlgebra[i] = O @ LieAlgebra[i] @ O.T
        
    # Define origin Euclidean vector x
    x = np.zeros((ambient_dim,1))
    for i in range(int(ambient_dim/2)): x[2*i,0] = 1
    x = O @ x
    x /= np.linalg.norm(x)

    # Draw sample from uniform distribution
    Sample = []
    Angles = [np.random.uniform(0,2*np.pi,n_points) for i in range(dim)]
    for i in range(n_points):
        s = x.copy()
        for j in range(dim): 
            coefficient = np.sqrt(2)*np.linalg.norm(frequencies[j])/ComputeGCD(frequencies[j])
            s = scipy.linalg.expm(Angles[j][i]*LieAlgebra[j]*coefficient) @ s
        Sample.append(s[:,0])
    Sample = np.array(Sample)
    
    return Sample, LieAlgebra




#sample on SO(3) orbits
def delta(i,j):
    '''
    Kronecker's delta function
    Input:
        -i: first coordinate
        -j: second coordinate
    Output:
        -value: 0 if i != j, else 1
    '''
    if i == j:
        return 1
    else: 
        return 0

def a_l(l,j):
    '''
    Coefficient a_l used in building the irreducible representation of SO(3)
    Input:
        -l: coordinate value 
        -j: representation type
    Output:
        -a_l^j
    '''
    return np.sqrt((2*l*j-l*(l-1))/4)

def rep_so_3(j):
    '''
    Makes the Lie algebra of irreducible real representations of SO(3)
    Input:
        - j: parameter of choosing irreducible representation type (dim rep = 2j+1)
    Output:
        - LieAlgebra: a list of length 3, containing (2j+1)x(2j+1) np matrices,
            the skew-symmetric matrices representing a derived Lie algebra basis
    '''
    X_1 = np.zeros((int(2*j)+1, int(2*j)+1))
    X_2 = np.zeros((int(2*j)+1, int(2*j)+1))
    X_3 = np.zeros((int(2*j)+1, int(2*j)+1))
    
    if type(j) == int:
        for k in range(1, 2*j+2):
            for l in range(1, 2*j+2):
                X_1[k-1,l-1] =  ((1+(-1)**k)/2)*(delta(l,k+1)*a_l(int(k/2),j)+delta(l+3,k)*a_l(int((k-2)/2),j))-(a_l(j,j)+np.sqrt((j**2+j)/2))*(delta(l,2*j+1)*delta(2*j,k)-delta(l,2*j)*delta(2*j+1,k))-((1+(-1)**(k-1))/2)*(delta(l,k+3)*a_l(int((k+1)/2),j)+delta(l+1,k)*a_l(int((k-1)/2),j))
                X_2[k-1,l-1] = -(a_l(j,j)+np.sqrt((j**2+j)/2))*(delta(l,2*j+1)*delta(2*j-1,k)-delta(l,2*j-1)*delta(2*j+1,k))+ delta(l,k+2)*a_l(int((k+1)/2),j)-delta(l+2,k)*a_l(int((k-1)/2),j)
                X_2[k-1,l-1] = - X_2[k-1,l-1]
                X_3[k-1,l-1] = 1/4*((1+(-1)**k)*delta(l+1,k)*(2*j+2-k)+((-1)**k-1)*delta(k+1,l)*(2*j+1-k))
    else:
        for k in range(1, int(2*j)+2):
            for l in range(1, int(2*j)+2):
                r = j
                X_1[k-1,l-1] = ((1+(-1)**(k-1))/2)*(delta(l,k+3)*a_l(int((k+1)/2),r)+delta(l+1,k)*a_l(int((k-1)/2),r))-((1+(-1)**k)/2)*(delta(l,k+1)*a_l(int(k/2),r)+delta(l+3,k)*a_l(int(k-2/2),r))
                X_2[k-1,l-1] = delta(l,k+2)*a_l(int((k+1)/2),r)-delta(l+2,k)*a_l(int((k-1)/2),r)
                X_3[k-1,l-1] = 1/4*((1+(-1)**k)*delta(l+1,k)*(2*j+2-k)+((-1)**k-1)*delta(k+1,l)*(2*j+1-k))
    return [X_1, X_2, X_3]

def SampleSO3(freq,  n_points = 500, conjugate = True, verbose = False, plot = False):
    '''
    Sample n point on an orbit of a representation of SO(3) of type given by the block sum of frequencies freq

    Input:
        - freq: a list of lists with frequencies of normal decomposition
        - n_points: number of points to sample
        - frequencymax: maximal frequency of the representation
        - conjugate: if False, will be a representation in the canonical basis
        - verbose: whether to print comments
        - plot: if True, plots a 3D PCA scatter of data is shown
    Output:
        - Sample: a (n_points)x(ambient_dim) np matrix, the sampled points on the orbit
        - LieAlgebra: a list of length dim, containing (ambient_dim)x(ambient_dim) np matrices,
            the skew-symmetric matrices representing a derived Lie algebra basis
        Example:
        X,LieAlgebra = datasets.SampleSO3([1],  conjugate = True, verbose = False, n_points = 5, plot = False)
        print(X)
        >>> [[ 0.64769745  1.28439777  0.96478515]
            [ 1.7143352   0.08829403 -0.23077908]
            [ 1.00251307  0.99528029  1.00218995]
            [ 0.75046591  1.27409457  0.90193345]
            [-0.36918789  1.18902584  1.20412535]]
        print(LieAlgebra[0])
        >>> [[-1.59400912e-17 -8.84614989e-01  3.99027824e-01]
            [ 8.84614989e-01 -3.38219836e-17  2.41315388e-01]
            [-3.99027824e-01 -2.41315388e-01 -1.30254707e-18]]
        print(LieAlgebra[1])
        >>> [[ 3.97022984e-18 -1.50251923e-01 -7.33785095e-01]
             [ 1.50251923e-01 -5.30373542e-18  6.62558521e-01]
             [ 7.33785095e-01 -6.62558521e-01 -2.11927710e-17]] 
        print(LieAlgebra[2])
        >>>  [[ 1.41142798e-18 -4.41452920e-01 -5.49851098e-01]
             [ 4.41452920e-01 -2.36531185e-18 -7.09071992e-01]
             [ 5.49851098e-01  7.09071992e-01  9.64403202e-18]]
    '''
    amb_dim = 0
    for j in freq: 
        amb_dim += 2*j+1
    A_1 = np.zeros((int(amb_dim), int(amb_dim))); A_2 = np.zeros((int(amb_dim), int(amb_dim))); A_3 = np.zeros((int(amb_dim), int(amb_dim)))
    counter = 0
    for j in freq:
        A_1[counter:2*j+1+counter,counter:2*j+1+counter],A_2[counter:2*j+1+counter,counter:2*j+1+counter], A_3[counter:2*j+1+counter,counter:2*j+1+counter] = rep_so_3(j)
        counter += 2*j+1
    
    if conjugate: 
        O = scipy.stats.special_ortho_group.rvs(int(amb_dim))
        A_1, A_2, A_3 = O @ A_1 @ O.T, O @ A_2 @ O.T, O @ A_3 @ O.T
    
    ts = np.linspace(0,2*np.pi,1000)
    Sample = []
    for N in range(n_points):
        t1,t2,t3 = random.choices(ts,k = 3)
        Sample.append(np.array(scipy.linalg.expm(A_1*t1 + A_2*t2 + A_3*t3)@np.ones(amb_dim)))
    Sample = np.array(Sample)
        
    if plot:
        velour.PlotPCA(Sample); plt.show();
    
    return Sample, [A_1,A_2, A_3]

