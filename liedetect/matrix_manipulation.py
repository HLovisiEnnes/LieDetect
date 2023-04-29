import autograd.numpy as np 
import itertools
import sympy
import scipy, sklearn #will be useful for the importation file later on

' Matrix manipulations '

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return list(itertools.chain.from_iterable(itertools.combinations(xs,n) for n in range(len(xs)+1)))

def ComputeGCD(integers):
    gcd = integers[0]
    for integer in integers[1:]: gcd = np.gcd(gcd,integer)
    return gcd

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)    

def SkewSymmetricToVector(A):
    '''
    Given a skew-symmetric matrix written in the canonical basis of M_n(R),
    writes it in the canonical basis of S_n(R)
    
    Example:
        A = np.array([[0,1,2],[-1,0,1], [-2,-1,0]]); print(A)
        >>> [[ 0  1  2]
             [-1  0  1]
             [-2 -1  0]] 
        v = SkewSymmetricToVector(A); print(v)
        >>> [1 2 1]
    '''
    d = np.shape(A)[0]
    
    # Skew-symmetrize the matrix, in case it is not skew-symmetric already
    As = (A-A.T)/2

    # Create the indices of the canonical basis of S_n(R)
    Indices = [tuple([i,j]) for i in range(d) for j in range(i+1,d)]
    
    # Get values of the indices corresponding to the basis
    Av = np.array([A[t] for t in Indices])    
    
    return Av

def VectorToSkewSymmetric(v): 
    '''
    Given a skew-symmetric matrix written in the canonical basis of S_n(R),
    writes it in the canonical basis of M_n(R)
    
    Example:
        v = np.array([1, 2, 1,]); print(v)
        >>> [1 2 1]
        B = VectorToSkewSymmetric(v); print(B)
        >>> [[ 0.  1.  2.]
             [-1.  0.  1.]
             [-2. -1.  0.]
    '''
    N = len(v)
    n = int((np.sqrt(1+8*N)+1)/2)
        
    # Create the indices of the canonical basis of S_n(R)
    Indices = [tuple([i,j]) for i in range(n) for j in range(i+1,n)]
    
    # Inverse the list Indices
    IndicesDicInv = {Indices[i]:i for i in range(len(Indices))}
    
    # Create a skew-symmetric matrix
    A = np.zeros((n,n))
    for ind in IndicesDicInv: A[ind] = v[IndicesDicInv[ind]]
    A -= A.T
    
    return A

def FrameToProjection(frame):
    '''
    Takes a list of n skew-symmetric matrices, representing a orthogonal 
    frame, and return a projection on this subspace, seen in S_n(R).
    
    Example:
        A = np.array([[0,1,2],[-1,0,1], [-2,-1,0]]); print(A)
        >>> [[ 0  1  2]
             [-1  0  1]
             [-2 -1  0]] 
        P = (FrameToProjection([A]); print(P)
        >>> [[0.16666667 0.33333333 0.16666667]
             [0.33333333 0.66666667 0.33333333]
             [0.16666667 0.33333333 0.16666667]]
    '''
    ambient_dim = np.shape(frame)[0]    
    
    # Convert matrices to vectors
    frame_vectors = [SkewSymmetricToVector(matrix) for matrix in frame]
    
    # Normalize the vectors
    frame_vectors_orth = [v/np.linalg.norm(v) for v in frame_vectors]
        
    # Apply Gram-Schmidt orthonormalization
    if len(frame)>1:
        for i in range(1,len(frame)):
            v = frame_vectors_orth[i]
            for j in range(i):
                w = frame_vectors_orth[j]
                v -= np.dot(v, w)*w
            v /= np.linalg.norm(v)
            frame_vectors_orth[i] = v
    
    # Define projection on the subspace (point in Grassmannian)
    Proj = np.sum([ np.outer(v,v) for v in frame_vectors_orth ],0)
    
    return Proj

def GramSchmidtOrthonormalizationVectors(Vectors):
    '''
    Orthonormalize a collection of vectors
    '''
    # Normalize the vectors
    Vectors = [A/np.linalg.norm(A) for A in Vectors]
        
    # Apply Gram-Schmidt orthonormalization
    if len(Vectors)>1:
        for i in range(1,len(Vectors)):
            v = Vectors[i]
            for j in range(i):
                w = Vectors[j]
                v -= np.dot(v, w)*w #scalar product of vectors
            v /= np.linalg.norm(v)
            Vectors[i] = v
            
    return Vectors

def GramSchmidtOrthonormalizationMatrices(Matrices):
    '''
    Orthonormalize a collection of matrices
    
    Example:
        Matrices = [np.array([[ 0.,  1.,  0.,  0.],[-1.,  0.,  0.,  0.],[ 0.,  0.,  0.,  2.],[ 0.,  0., -2.,  0.]]),
                    np.array([[ 0.,  1.,  0.,  0.],[-1.,  0.,  0.,  0.],[ 0.,  0.,  0.,  1.],[ 0.,  0., -1.,  0.]])]
        GramSchmidtOrthonormalizationMatrices(Matrices)
        >>> [array([[ 0.        ,  0.31622777,  0.        ,  0.        ],
                    [-0.31622777,  0.        ,  0.        ,  0.        ],
                    [ 0.        ,  0.        ,  0.        ,  0.63245553],
                    [ 0.        ,  0.        , -0.63245553,  0.        ]]),
             array([[ 0.        ,  0.63245553,  0.        ,  0.        ],
                    [-0.63245553,  0.        ,  0.        ,  0.        ],
                    [ 0.        ,  0.        ,  0.        , -0.31622777],
                    [ 0.        ,  0.        ,  0.31622777,  0.        ]])]
    '''
    # Normalize the vectors
    Matrices = [A/np.linalg.norm(A) for A in Matrices]
        
    # Apply Gram-Schmidt orthonormalization
    if len(Matrices)>1:
        for i in range(1,len(Matrices)):
            v = Matrices[i]
            for j in range(i):
                w = Matrices[j]
                v -= np.sum(np.multiply(v, w))*w #scalar product of matrices
            v /= np.linalg.norm(v)
            Matrices[i] = v
            
    return Matrices

'Lie Algebra manipulations '

def GetCanonicalLieAlgebraTorus(frequencies):
    '''
    Convert a list of list of frequencies to a list of skew-symmetric matrices,
    the canonical matrices representing the Lie algebra of a torus representation.
    
    Example:
        L = GetCanonicalLieAlgebraTorus([[1,3]]); print(L)
        >>> [array([[ 0.,  1.,  0.,  0.],
                    [-1.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  3.],
                    [ 0.,  0., -3.,  0.]])]
    '''
    # Create basis
    m = len(frequencies[0])
    Basis = [np.zeros((2*m,2*m)) for i in range(m)]
    for i in range(m): Basis[i][2*i,2*i+1], Basis[i][2*i+1,2*i]= 1, -1
        
    # Generate canonical rotations for the angles
    CanonicalLieAlgebra = [np.sum([Basis[j]*frequencies[i][j] for j in range(m)],0) for i in range(len(frequencies))]
        
    return CanonicalLieAlgebra

def GetFrequenciesToTest(dim, ambient_dim, frequency_max):
    '''
    Gives a list of the frequencies of all the representations of the torus of 
    dimension dim, in the ambient space R^(ambient_dim), whose frequencies are
    at most frequency_max.
    
    Input:
        - dim: dimension of the torus, must be equal to len(frequencies)
        - ambient_dim: dimension of the ambient space
        - frequency_max: maximal frequency value
        
    Output:
        - FrequenciesToTest: a list of tuples of tuples
    
    Example: 
        F = GetFrequenciesToTest(2, 4, 2); print(F)
        >>> [((1, 1), (1, 2)), ((1, 2), (2, 1)), ((1, 2), (2, 2))]
    '''
    m = int(ambient_dim/2)

    # Initial list of frequencies: Cartesian product
    Frequencies = list(itertools.product(range(1,frequency_max+1), repeat=m))
    FrequenciesToTest = list(itertools.combinations(Frequencies,dim))

    # Remove frequencies with only one value
    FrequenciesToTest = [freq for freq in FrequenciesToTest if len(set([i for f in freq for i in f]))>1]

    # Remove frequencies with a common denominator
    # --> test if GCD is 1
    FrequenciesToTest = [freq for freq in FrequenciesToTest if ComputeGCD([ComputeGCD(f) for f in freq])<2]
    
#     # Remove duplicates up to permutation of the components
#     # --> put the smallest (in lexicographic order) first
#     FrequenciesToTest = [tuple(sorted(freq)) for freq in FrequenciesToTest]
#     FrequenciesToTest = list(set(FrequenciesToTest))

    # Remove duplicates up to permutation of the components
    # --> put the smallest (according to the sum) first
    FrequenciesToTest1 = []
    for freq in FrequenciesToTest:
        lengths=[np.sum(f) for f in freq]
        freq = tuple([freq[i] for i in np.argsort(lengths)])
        FrequenciesToTest1.append(freq)
    FrequenciesToTest=FrequenciesToTest1

    # Remove duplicates up to permutation of the integers
    # --> sort the first component
    FrequenciesToTest_unique = []
    for freq in FrequenciesToTest:
        tuples = [[freq[i][j] for i in range(dim)] for j in range(m)]
        indices = argsort(tuples)
        freqsorted = tuple([tuple([f[i] for i in indices]) for f in freq])
        FrequenciesToTest_unique.append( freqsorted )
    FrequenciesToTest = list(set(FrequenciesToTest_unique))

    # Only linearly independant - Remove frequencies with repeated tuples
    # --> sort the first component
    FrequenciesToTest_unique = []
    for freq in FrequenciesToTest:
        tuples = [tuple([freq[i][j] for i in range(dim)]) for j in range(m)]
        tuples = list(set(tuples))
        if len(tuples)==m: # no repeated tuples
            FrequenciesToTest_unique.append( freq )
    FrequenciesToTest = list(set(FrequenciesToTest_unique))
    
    # Remove tuples that span the same space
    # --> compute rref
    rrefs = {freq:tuple(sympy.Matrix(freq).rref()[0]) for freq in FrequenciesToTest}
    rrefs_inv = {rrefs[freq]:freq for freq in rrefs}
    FrequenciesToTest = list(rrefs_inv.values())

    # Sort the final list to ease the reading
    FrequenciesToTest = sorted(FrequenciesToTest)
    
    return FrequenciesToTest

def NormalFormSkewSymmetricMatrix(A):
    '''
    Compute the frequencies of the invariant planes of a skew-symmetric matrix A.
    
    Input:   
        - A: a nxn numpy matrix
    Output:  
        - Frequencies: a list of length floor(n/2)
        - T: the block-diagonal matrix
        - Z: the change of basis (orthogonal)
    '''
    m = int(np.shape(A)[0]/2)    
    
    # Project A on the skew-symmetric matrices
    As = (A-A.T)/2
    
    # Compute the Schur decomposition of A
    T , Z = scipy.linalg.schur(As) # we have Z @ T @ Z.T = As
    
    # Set off-block values to exactly 0
    index = [tuple([2*i+1, 2*i]) for i in range(m)] + [tuple([2*i, 2*i+1]) for i in range(m)]
    T2 = T*0
    for t in index: T2[t] = T[t]
    T = T2 
    
    # Extract the frequencies of the invariant planes
    Frequencies = [T[i,i+1] for i in range(0,np.shape(A)[0],2)]
    
    # Normalize the frequencies
    Frequencies /= np.linalg.norm(Frequencies)
    
    # Find negative frequencies
    index = np.where(Frequencies<0)[0]
    
    # Make frequencies positive
    for i in index:
        Frequencies[i] = - Frequencies[i]
        T[2*i:2*i+2,2*i:2*i+2] *= -1
        Z[:,2*i+1] *= -1
    
    # Sort frequencies    
    index = np.argsort(Frequencies)
    Frequencies = Frequencies[index]
    index_zip = [item for sublist in zip(2*index, 2*index+1) for item in sublist]
    T = T[:,index_zip]; T = T[index_zip,:]
    Z = Z[:,index_zip]
    
    # Sanity check
    if np.linalg.norm(Z @ T @ Z.T - As)>1e-10: 
        print('Error in NormalFormSkewSymmetricMatrix! norm = ', np.linalg.norm(Z @ T @ Z.T - As))
    return Frequencies, Z, T