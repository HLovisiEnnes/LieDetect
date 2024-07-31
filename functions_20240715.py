'''----------------------------------------------------------------------------

LieDetect
Henrique Ennes & Raphaël Tinarrage
Last update: 22/07/2024

See the repo at https://github.com/HLovisiEnnes/LieDetect and the article at https://arxiv.org/abs/2309.03086

Misc: 
- ChronometerStart
- ChronometerStop
- ChronometerTick

Matrix manipulations:
- powerset
- ComputeGCD
- argsort
- SkewSymmetricToVector
- VectorToSkewSymmetric
- FrameToProjection
- GramSchmidtOrthonormalizationVectors
- GramSchmidtOrthonormalizationMatrices

Lie Algebra manipulations:
- GetCanonicalLieAlgebraTorus
- ProjectLatticeOnQuadrant
- AreFrequencesEquivalent
- GetPrimitiveVectors
- GetRandomLattice
- GetPrimitiveLattices
- GetFrequenciesToTest
- NormalFormSkewSymmetricMatrix
- NormalFormSkewSymmetricMatrices
- partition
- partition_so3
- partition_su2
- rep_su2
- GetCoefficientSU2
- GetCanonicalLieAlgebraSU2

Optimizations and Lie-PCA:
- PrintCovarianceEigenvalues
- Orthonormalize
- PlotNormsDistribution
- GetLiePCAOperator
- FindClosestLieAlgebra
- OptimizationGrassmann
- OptimizationStiefel
- GenerateOrbitTorus
- GenerateOrbitSU2

Functions to sample points:
- SampleOnTorus
- SampleOnSU2

----------------------------------------------------------------------------'''

import autograd.numpy as np 
from autograd import grad   
import matplotlib.pyplot as plt
import pymanopt, ot, scipy, skdim, sklearn
import math, functools, copy, time, datetime, random, itertools, sys, IPython

'''----------------------------------------------------------------------------
Misc
----------------------------------------------------------------------------'''

def ChronometerStart(msg='Start... '):
    start_time = time.time()
    sys.stdout.write(msg); sys.stdout.flush()
    return start_time
    
def ChronometerStop(start_time, method='ms', linebreak='\n'):
    elapsed_time_secs = time.time() - start_time
    if method == 'ms':
        msg = 'Execution time: '+repr(round(elapsed_time_secs*1000))+' ms.'+linebreak
    if method == 's':
        msg = 'Execution time: '+repr(round(elapsed_time_secs))+' s.'+linebreak
    sys.stdout.write(msg); sys.stdout.flush()    
    
def ChronometerTick(start_time, i, i_total, msg):    
    elapsed_time_secs = time.time() - start_time
    expected_time_secs = (i_total-i-1)/(i+1)*elapsed_time_secs
    msg1 = 'It '+repr(i+1)+'/'+repr(i_total)+'. '
    msg2 = 'Duration %s ' % datetime.timedelta(seconds=round(elapsed_time_secs))
    msg3 = 'Expected remaining time %s.' % datetime.timedelta(seconds=round(expected_time_secs))
    sys.stdout.write('\r'+msg+msg1+msg2+msg3)
    if i>=i_total-1: sys.stdout.write('\n')
    
'''----------------------------------------------------------------------------
Matrix manipulations
----------------------------------------------------------------------------'''

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
    A = (A-A.T)/2

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
    
'''----------------------------------------------------------------------------
Lie Algebra manipulations
----------------------------------------------------------------------------'''

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

def ProjectLatticeOnQuadrant(decirrep):
    decirreppositive = [tuple([x*val for x in irr]) for irr in decirrep for val in [np.sign(next((x for i, x in enumerate(irr) if x!=0), None))]]
    decirreppositive = tuple(sorted(decirreppositive))
    return decirreppositive

def AreFrequencesEquivalent(freq0,freq1,verbose=True):    
    if type(freq0)==tuple: # case of SO(2) or torus frequencies
        dim, m = len(freq0), len(freq0[0])
        # Convert to decomposition into irreps
        decirrep0, decirrep1 = tuple(list(zip(*freq0))), tuple(list(zip(*freq1)))

        Projections0, Projections1 = [], []
        Permutations = itertools.permutations(range(m))
        Signs = itertools.product([-1,1],repeat=m)
        SignedPermutations = list(itertools.product(Permutations,Signs))
        for (perm, signs) in SignedPermutations:
            # Permute irreps
            decirrep0perm = tuple([tuple([decirrep0[perm[i]][j]*signs[i] for j in range(dim)]) for i in range(m)])
            decirrep1perm = tuple([tuple([decirrep1[perm[i]][j]*signs[i] for j in range(dim)]) for i in range(m)])

            # Compute projections
            frame_vectors_orth = GramSchmidtOrthonormalizationVectors(np.array(decirrep0perm).T)
            Projections0 += [np.sum([ np.outer(v,v) for v in frame_vectors_orth ],0).flatten()]
            frame_vectors_orth = GramSchmidtOrthonormalizationVectors(np.array(decirrep1perm).T)
            Projections1 += [np.sum([ np.outer(v,v) for v in frame_vectors_orth ],0).flatten()]

        # Compute distances
        dist = sklearn.metrics.pairwise_distances(Projections0,Projections1)
        AreEquivalent = np.min(dist)<1e-5

    elif type(freq0)==list: # case of SO(3) or SU(2) frequencies
        AreEquivalent = list(freq0)==list(freq1)

    if verbose:
        if AreEquivalent: print('----> The frequencies', freq0, 'and', freq1, '\x1b[1;31m'+'are equivalent'+'\x1b[0m', '<----')
        else: print('----> The frequencies', freq0, 'and', freq1, '\x1b[1;31m'+'are not equivalent'+'\x1b[0m', '<----')        
    return AreEquivalent

def GetPrimitiveVectors(m, frequency_max):
    FrequenciesToTest = itertools.combinations(range(1,frequency_max+1), m)
    FrequenciesToTest = [(tuple(v),) for v in FrequenciesToTest if ComputeGCD(v)<2]
    return FrequenciesToTest

def GetRandomLattice(dim, ambient_dim, frequency_max, verbose=0):
    '''
    Generate a random lattice, without repetition of the irreps
    '''
    m = int(ambient_dim/2)
    # Generate irreps
    Irreps = list(itertools.product(range(0,frequency_max+1), repeat=dim))
    Irreps.remove(tuple([0]*dim))
    Irreps = [irr for irr in Irreps if irr[next((i for i, x in enumerate(irr) if x), None)]>0]
        # only irreps with positive first nonzero value
    # Pick random sample of irreps
    SampleIrreps = random.sample(Irreps, m)
    # Multiply by 1 or -1 the weights
    Coefficient = random.choices([-1,1], k=m)
    SampleIrreps = tuple([tuple([SampleIrreps[i][j]*Coefficient[i] for j in range(dim)]) for i in range(m)])
    # Transpose to get lattice
    lattice = tuple(list(zip(*SampleIrreps)))
    return lattice

def GetPrimitiveLattices(dim, ambient_dim, frequency_max, method='repetitions', verbose=0):
    '''
    Gives a list the primitive lattices of dimension dim in R^{2*ambient_dim}.
    method can be 'exact' (only one lattice per equivalence class) or 'repetitions' (potentially repetitions)
    '''
    m = int(ambient_dim/2)

    # Initial list of frequencies: Cartesian product
    if verbose>0: start_time = ChronometerStart('Decomposition into irreps... ')
    # Irreps = list(itertools.product(range(-frequency_max,frequency_max+1), repeat=dim))
    Irreps = list(itertools.product(range(0,frequency_max+1), repeat=dim))
    Irreps.remove(tuple([0]*dim))
    Irreps = [irr for irr in Irreps if irr[next((i for i, x in enumerate(irr) if x), None)]>0]
        # only irreps with positive first nonzero value
    Lattices = list(itertools.combinations(Irreps,m))
        # tuples of distinct irreps (because faithful representation)
    Lattices = sorted(Lattices)
    if verbose>0: ChronometerStop(start_time, 's')
    if verbose>0: print('Number of equivalence classes of reps:', len(Lattices))
        
    # Preprocess: discard not maximak rank
    if verbose>0: start_time = ChronometerStart('Get rank (preprocess)... ')
    IndicesFullRank = [i for i in range(len(Lattices)) if np.linalg.matrix_rank(np.array(Lattices[i]).T)==dim]
    if verbose>0: ChronometerStop(start_time, 's')
    if verbose>1: print('Discarded because not maximal rank (preprocess):', len(Lattices)-len(IndicesFullRank), 'out of', len(Lattices))
    Lattices = [Lattices[i] for i in IndicesFullRank]
    Lattices = sorted(Lattices)

    if method == 'repetitions':
        FrequenciesToTest = [tuple(list(zip(*lattice))) for lattice in Lattices]
        if verbose>0: print('Number of frequencies to test:', len(FrequenciesToTest))
        return FrequenciesToTest

    if method == 'exact':
        # Preprocess
        epsilon = 1e-5
        Permutations = itertools.permutations(range(m))
        Signs = itertools.product([-1,1],repeat=m)
        SignedPermutations = list(itertools.product(Permutations,Signs))
        Projections = {lattice:[] for lattice in Lattices}
        if verbose>0: ChronometerStart('Get projection matrices... ')
        for decirrep in Lattices:
            for (perm, signs) in SignedPermutations:
                # Permute irreps
                decirrepperm = tuple([tuple([decirrep[perm[i]][j]*signs[i] for j in range(dim)]) for i in range(m)])
                frame_vectors_orth = GramSchmidtOrthonormalizationVectors(np.array(decirrepperm).T)
                Projections[decirrep] += [np.sum([ np.outer(v,v) for v in frame_vectors_orth ],0).flatten()]
        if verbose>0: ChronometerStop(start_time, 's')
        if verbose>0: print('Number of bases via signed permutation:', len(Lattices)*len(SignedPermutations))
            
        # Compare matrices
        epsilon = 1e-5
        LatticesToTest = [True for lattice in Lattices]
        NewLattices = []
        while any(LatticesToTest):
            newindex = next((i for i, x in enumerate(LatticesToTest) if x), None)
            lattice = Lattices[newindex]
            NewLattices.append(lattice)
            LatticesToTest[newindex] = False
            IndicesToCompare = [i for i in range(len(LatticesToTest)) if LatticesToTest[i]==True]
            for i in IndicesToCompare:
                dist = sklearn.metrics.pairwise_distances(Projections[lattice],Projections[Lattices[i]])
                if np.min(dist)<epsilon:
                    LatticesToTest[i] = False
        
        FrequenciesToTest = [tuple(list(zip(*lattice))) for lattice in NewLattices]
        if verbose>0: print('Number of frequencies to test:', len(FrequenciesToTest))
        return FrequenciesToTest

def GetFrequenciesToTest(dim, ambient_dim, frequency_max,verbose=0):
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
    if dim==1: 
        m = int(ambient_dim/2)
        FrequenciesToTest = GetPrimitiveVectors(m,frequency_max)
    else:
        FrequenciesToTest = GetPrimitiveLattices(dim, ambient_dim, frequency_max, verbose=False)
    if verbose>0: print('Number of frequencies to test:', len(FrequenciesToTest))
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

def NormalFormSkewSymmetricMatrices(Matrices, verbose=True, verbosity=0):
    d = np.shape(Matrices[0])[0]
    m = int(np.shape(Matrices[0])[0]/2)

    # Optimization with pymanopt
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(d, k=1)

    @pymanopt.function.autograd(manifold)
    def cost(O):
        # Conjugate canonical matrices
        MatricesConjugate = [O @ A @ O.T for A in Matrices]
        
        # Compute norm block-anti-diagonal terms
        Terms = []
        for i in range(len(Matrices)): 
            for j in range(m):
                Terms += [ MatricesConjugate[i][2*j+0,2*j+1], MatricesConjugate[i][2*j+1,2*j+0] ]
        Terms = np.array(Terms)
        
        # Compute norm
        differences = np.array([np.trace(A @ A.T) for A in MatricesConjugate])
        return np.sum(differences @ differences.T) - np.sum(Terms @ Terms.T)
    
    # Run optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result = optimizer.run(problem)
    cost, O = result.cost, result.point
    
    # Transform in normal form
    MatricesConjugate = [O @ A @ O.T for A in Matrices]
    frequencies = tuple([tuple([MatricesConjugate[i][2*j+0,2*j+1] for j in range(m)]) for i in range(len(Matrices))])
        
    return frequencies, O

def partition(n):
    '''
    https://jeromekelleher.net/generating-integer-partitions.html
    '''
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]
        
def partition_so3(n):
    '''
    Gets the partitions of only odd terms of some n
    '''
    parts = []
    for part_list in partition(n):
        fl = 1
        for j in part_list:
            if j%2 == 0:
                fl = 0
                break
        if fl:
            parts.append(part_list)
    parts.remove([1]*n) #remove the trivial representation
    return parts  

def partition_su2(n):
    '''
    Gets the partitions of some n, excluded even terms not divisible by 4
    '''
    parts = []
    for part_list in partition(n):
        fl = 1
        for j in part_list:
            if j%4 == 2:
                fl = 0
                break
        if fl:
            parts.append(part_list)
    parts.remove([1]*n) #remove the trivial representation
    return parts 

def rep_su2(n, conjugate = False):
    def delta(i,j): return (i==j)*1    
    def a_l(l,j): return np.sqrt((2*l*j-l*(l-1))/4)
    
    if n==1: j=0
    elif n%2==1: j=int((n-1)/2)
    elif n%4==0: j = (n-2)/4
    else: 
        print('Error:', n, 'is not a dimension of an irrep of SU(2).')
        return None
    
    if type(j) == int:
        X_1 = np.zeros((int(2*j)+1, int(2*j)+1))
        X_2 = np.zeros((int(2*j)+1, int(2*j)+1))
        X_3 = np.zeros((int(2*j)+1, int(2*j)+1))
        for k in range(1, 2*j+2):
            for l in range(1, 2*j+2):
                X_1[k-1,l-1] =  ((1+(-1)**k)/2)*(delta(l,k+1)*a_l(int(k/2),j)+delta(l+3,k)*a_l(int((k-2)/2),j))-(a_l(j,j)+np.sqrt((j**2+j)/2))*(delta(l,2*j+1)*delta(2*j,k)-delta(l,2*j)*delta(2*j+1,k))-((1+(-1)**(k-1))/2)*(delta(l,k+3)*a_l(int((k+1)/2),j)+delta(l+1,k)*a_l(int((k-1)/2),j))
                X_2[k-1,l-1] = -(a_l(j,j)+np.sqrt((j**2+j)/2))*(delta(l,2*j+1)*delta(2*j-1,k)-delta(l,2*j-1)*delta(2*j+1,k))+ delta(l,k+2)*a_l(int((k+1)/2),j)-delta(l+2,k)*a_l(int((k-1)/2),j)
                X_2[k-1,l-1] = - X_2[k-1,l-1]
                X_3[k-1,l-1] = 1/4*((1+(-1)**k)*delta(l+1,k)*(2*j+2-k)+((-1)**k-1)*delta(k+1,l)*(2*j+1-k))
    else:
        X_1 = np.zeros((int(4*j)+2, int(4*j)+2))
        X_2 = np.zeros((int(4*j)+2, int(4*j)+2))
        X_3 = np.zeros((int(4*j)+2, int(4*j)+2))
        for k in range(1, int(4*j)+3):
            for l in range(1, int(4*j)+3):
                r = j
                X_1[k-1,l-1] = ((1+(-1)**(k-1))/2)*(delta(l,k+3)*a_l(int((k+1)/2),r)+delta(l+1,k)*a_l(int((k-1)/2),r))-((1+(-1)**k)/2)*(delta(l,k+1)*a_l(int(k/2),r)+delta(l+3,k)*a_l(int((k-2)/2),r))
                X_2[k-1,l-1] = delta(l,k+2)*a_l(int((k+1)/2),r)-delta(l+2,k)*a_l(int((k-1)/2),r)
                X_3[k-1,l-1] = 1/4*((1+(-1)**k)*delta(l+1,k)*(2*j+2-k)+((-1)**k-1)*delta(k+1,l)*(2*j+1-k))
    if conjugate: 
        O = scipy.stats.special_ortho_group.rvs(2*int(j)+1)
        X_1 = O @ X_1 @ O.T
        X_2 = O @ X_2 @ O.T
        X_3 = O @ X_3 @ O.T
    return X_1, X_2, X_3

def GetCoefficientSU2(A):
    '''
    Return a t>0 such that exp(tA)=I. Works only for skew-matrices matrices with 
    integer of half-integer frequencies, such as those returned by rep_su2.
    '''
    # Compute the Schur decomposition of A
    As = (A-A.T)/2
    if np.linalg.norm(As-A)>1e-5: print('Error! Matrix is not skew-symmetric')
    T , Z = scipy.linalg.schur(As) # we have Z @ T @ Z.T = As
    Frequencies = [T[i,i+1] for i in range(np.shape(A)[0]-1)]

    # Get the integral frequencies
    Frequencies = [2*f for f in Frequencies]
        #multiply by 2, because half-integer frequencies for half-integer parameter
    Frequencies = sorted([np.abs(f) for f in Frequencies if np.abs(f)>1e-1])
        #only nonzero frequencies
    if any([np.abs(f-np.rint(f))>1e-5 for f in Frequencies]): print('Error! Frequencies not integers', Frequencies)
    Frequencies = [int(np.rint(f)) for f in Frequencies]
        #convert to integers

    # Define the coefficient
    t = 4*np.pi/ComputeGCD(Frequencies)
    if np.linalg.norm(scipy.linalg.expm(t*A)-np.eye(np.shape(A)[0]))>1e-4: print('Error! Incorrect coefficient. Distance to identity:', np.linalg.norm(scipy.linalg.expm(t*A)-np.eye(np.shape(A)[0])))
        #sanity check
    return t

def GetCanonicalLieAlgebraSU2(partition):
    '''
    Convert a list of list of frequencies to a list of skew-symmetric matrices,
    the canonical matrices representing the Lie algebra of a SU(2) representation.
    '''
    n, m = sum(partition), len(partition)
    LieAlgebras = [rep_su2(k) for k in partition]
    CanonicalLieAlgebra = [np.zeros((n,n)) for i in range(3)]
    index = 0
    for i in range(m):
        k = partition[i]
        for j in range(3):
            CanonicalLieAlgebra[j][index:(index+k),index:(index+k)] = LieAlgebras[i][j]
        index+=k        
    return CanonicalLieAlgebra
    
'''----------------------------------------------------------------------------
Optimizations and Lie-PCA
----------------------------------------------------------------------------'''

def PrintCovarianceEigenvalues(X):
    S = np.mean([np.outer(X[i,:],X[i,:]) for i in range(np.shape(X)[0])],0)
    eigenvalues = np.flip(sorted(np.linalg.eig(S)[0]))
    eigenvalues = eigenvalues/np.sum(eigenvalues)
    print('Covariance eigenvalues:', [round(e,3) for e in eigenvalues])

def Orthonormalize(X):
    # Center
    X -= np.mean(X,0)
    
    # Homogeneization
    Cov = np.mean([np.outer(X[i,:],X[i,:]) for i in range(np.shape(X)[0])],0)
    Cov = scipy.linalg.sqrtm(np.linalg.inv(Cov))
    X = np.array([Cov.dot(x) for x in X])
    X = np.real(X) # in case X has complex values, because of numerical errors in PCA

    # Normalize
    mean_norm = np.mean(np.linalg.norm(X, axis=1))
    X /= mean_norm
    Cov /= mean_norm

    return X, Cov
    
def PlotNormsDistribution(X):
    plt.figure(figsize=(8,1));
    plt.hist([np.linalg.norm(p) for p in X]); 
    plt.title('Distribution of norms');

def GetLiePCAOperator(X,n_neighbors,dim=None,method='localPCA',correction=False,verbose=True):
    '''
    The method can be 'localcovariance' or 'localPCA'
    The correction refers to the reduction of the operator to the skew-symmetric matrices,
        and such that it is the identity on the symmetric matrices.
    '''
    if verbose==True: print('----> Lie PCA with', method, 'estimation of tangent spaces <----')
    ambient_dim = np.shape(X)[1]
    
    # Compute projection on tangent spaces
    kdt = sklearn.neighbors.KDTree(X, leaf_size=n_neighbors+1, metric='euclidean')
    neighbors_indices = kdt.query(X, n_neighbors+1, return_distance=False)[:,1::]
    ProjTangentSpaces = [np.sum([np.outer(X[i, :] - X[neighbors_indices[i,j], :],X[i, :] - X[neighbors_indices[i,j], :]) for j in range(n_neighbors)],0)
                               for i in range(np.shape(X)[0])]                                 #local covariance matrices
    ProjTangentSpaces = [(proj+proj.T)/2 for proj in ProjTangentSpaces]                        #symmetrize
    ProjTangentSpaces = [proj/np.linalg.norm(proj)*np.sqrt(dim) for proj in ProjTangentSpaces] #normalize
    
    if method=='localcovariance':
        # Compute normal spaces - robust
        ProjNormalSpaces = [np.eye((ambient_dim))-proj for proj in ProjTangentSpaces] #complementary
        
    if method=='localPCA':
        # Compute normal spaces - eigencut
        ProjNormalSpaces = []
        for i in range(np.shape(X)[0]):
            proj = ProjTangentSpaces[i]
            eigenvalues , Z = scipy.linalg.eigh(proj) 
            Tproj = np.eye(ambient_dim)
            for j in range(dim):
                index = np.argsort(eigenvalues)[-(j+1)] #the ith largest eigenvalue
                Tproj[index,index] = 0
            ProjNormalSpaces.append( Z @ Tproj @ Z.T )        

    # Compute projection on lines
    ProjLines = [np.outer(X[i,:],X[i,:])/np.dot(X[i,:],X[i,:]) for i in range(np.shape(X)[0])]

    # Create basis
    Basis = []
    for i in range(ambient_dim):
        for j in range(ambient_dim):
            basis = np.zeros((ambient_dim,ambient_dim))
            basis[i,j] = 1
            Basis.append(basis)

    # Compute Lie-PCA operator
    Sigma = np.zeros((ambient_dim**2,ambient_dim**2))
    Images = [np.sum([ProjNormalSpaces[j] @ Basis[i] @ ProjLines[j] for j in range(np.shape(X)[0])],0) for i in range(len(Basis))]
    for i in range(len(Basis)): Sigma[:,i] = Images[i].flatten()
    Sigma /= len(X)

    if correction==True:
        # Correction Lie-PCA operator
        Sigma_corrected = np.zeros((ambient_dim**2,ambient_dim**2))
        for k in range(len(Basis)):
            M = Basis[k]
            M_sym =     (M+M.T)/2
            M_skewsym = (M-M.T)/2
            M_sym_image, M_skewsym_image = M_sym, (Sigma@(M_skewsym.reshape(-1))).reshape(ambient_dim,ambient_dim)
            M_image = M_sym_image + M_skewsym_image
            Sigma_corrected[:,k] = M_image.flatten()
        Sigma_corrected = (Sigma_corrected+Sigma_corrected.T)/2
        Sigma = Sigma_corrected

    if verbose==True: 
        # Get eigenvectors
        vals, vecs = np.linalg.eig(Sigma) #finds eigenvalues and vectors of sigma as a matrix
        vals = np.real(vals)
        indices = np.argsort(vals)
        vals, vecs = [vals[i] for i in indices], list([vecs[:,i] for i in indices])
        print('First eigenvalues:', vals[0:4])
        print('Eigengap #'+str(dim)+':', vals[dim]/vals[dim-1])
        
    return Sigma

def FindClosestLieAlgebra(group, Sigma, dim=None, ambient_dim=4, frequency_max=3, FrequenciesToTest=None, method='Grassmann',verbosity=1,return_ChangeOfBasis=False):
    '''
    Input: 
    - group: can be 'torus', 'SU(2)' or 'SO(3)'
    - dim: dimension of the torus, if group='torus', and 3 if group='SU(2)' or 'SO(3)'
    - method: can be 'Grassmann' or 'Stiefel' or 'NormalForm' (the latter only for group='torus')
    - verbosity: 0 (no verbose), 1 (a few comments), 2 (comments from pymanopt)
    '''
    if verbosity>0: print('----> Optimization problem via'+'\x1b[1;31m', method, '\x1b[0m'+'for', group, 'group <----')

    # Get FrequenciesToTest
    if FrequenciesToTest==None:
        if group=='torus': FrequenciesToTest = GetFrequenciesToTest(dim=dim, ambient_dim=ambient_dim, frequency_max=frequency_max, verbose=verbosity)
        if group=='SU(2)': FrequenciesToTest = [tuple(f) for f in partition_su2(ambient_dim)]
        if group=='SO(3)': FrequenciesToTest = [tuple(f) for f in partition_so3(ambient_dim)]

    if method in ['Grassmann', 'NormalForm']:
        # Get and sort eigenvectors of Sigma
        vals, vecs = np.linalg.eig(Sigma) # finds eigenvalues and vectors of sigma as a matrix
        vecs = [vecs[:,i] for i in range(len(vecs))]
        indices = np.argsort(vals)        # argsort eigenvalues
        vals, vecs = [np.real(vals[i]) for i in indices], [np.real(vecs[i]) for i in indices]

        # Get bottom eigenvectors of Sigma
        LieAlgebra = [vecs[i].reshape((ambient_dim,ambient_dim)) for i in range(dim)]
        LieAlgebra = [(A-A.T)/2 for A in LieAlgebra]

    if method in ['Grassmann']:
        # Define Projection0, the projection matrix on the vector subspace given by Lie PCA
        Projection0 = FrameToProjection(LieAlgebra)
        
    if method in ['Grassmann', 'Stiefel']:
        # Run optimization with pymanopt
        if verbosity>0: start_time = ChronometerStart('Solve minimization problem... ')
        Costs = dict() 
        ChangeOfBasis = dict()
        for i in range(len(FrequenciesToTest)):
            frequencies = FrequenciesToTest[i]
            Costs[frequencies] = np.inf
            for determinant in ['+1','-1']:
                if method=='Grassmann': result = OptimizationGrassmann(group, Projection0, frequencies, ambient_dim, dim, determinant, verbosity=0)    
                elif method=='Stiefel': result = OptimizationStiefel(group, Sigma, frequencies, ambient_dim, dim, determinant, verbosity=0)  
                if result.cost < Costs[frequencies]: Costs[frequencies], ChangeOfBasis[frequencies] = result.cost, result.point
                if verbosity>2: print('Frequencies', frequencies, '- det', determinant, '- result', result.cost)
                if verbosity>3: print('('+result.stopping_criterion+')')
            if verbosity>0: ChronometerTick(start_time, i, len(FrequenciesToTest), 'Solve minimization problem... ')

        # Define optimal angle and cost
        OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
        OptimalChangeOfBasis = ChangeOfBasis[OptimalFrequencies]
            
        # Define OptimalLieAlgebra
        if group=='torus': OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
        if group=='SU(2)': OptimalLieAlgebra = GetCanonicalLieAlgebraSU2(OptimalFrequencies)
        OptimalLieAlgebra = [OptimalChangeOfBasis @ OptimalLieAlgebra[i] @ OptimalChangeOfBasis.T for i in range(dim)]
        OptimalLieAlgebra = GramSchmidtOrthonormalizationMatrices(OptimalLieAlgebra)
    
    elif method in ['NormalForm']:
        if dim==1:    
            # Estimate frequencies
            angles, OptimalChangeOfBasis, T = NormalFormSkewSymmetricMatrix(LieAlgebra[0])
            angles = np.sort(np.abs(angles))
            if verbosity>1: print('Normalized frequencies found in normal form decomposition:', tuple(angles))

            # Generate frequencies to test
            FrequenciesToTestNorm = [list(f[0])/np.linalg.norm(f[0]) for f in FrequenciesToTest]

            # Find closest frequencies
            Costs = {FrequenciesToTest[i]:np.linalg.norm(FrequenciesToTestNorm[i]-angles) for i in range(len(FrequenciesToTest))} 
            OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
                                                    
            # Define Lie algebra
            L = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
            OptimalLieAlgebra = [OptimalChangeOfBasis @ L[0]/np.linalg.norm(L[0]) @ OptimalChangeOfBasis.T]
            
        else: #dim >1
            # Orthonormalize 
            LieAlgebra = GramSchmidtOrthonormalizationMatrices(LieAlgebra)
            if verbosity>1: print('Test commutation:', np.linalg.norm(LieAlgebra[0] @ LieAlgebra[1] - LieAlgebra[1] @ LieAlgebra[0]))

            # Get frequencies via normal form
            frequencies, OptimalChangeOfBasis = NormalFormSkewSymmetricMatrices(LieAlgebra)
            if verbosity>1: print('Normalized frequencies found in normal form decomposition:', tuple(frequencies))
            
            # Expand GetFrequenciesToTest by permutation and negative values
            m = int(ambient_dim/2)
            indices = powerset(range(m))
            FrequenciesToTest_dic = {} #dict to remember the initial value of freq
            # 1/2: Put negative values
            FrequenciesToTest_positivenegative = []  #expanded frequencies
            for freq in FrequenciesToTest:
                freqlist = [list(f) for f in freq]
                for i in range(len(indices)):
                    freqi = copy.deepcopy(freqlist)
                    for j in indices[i]:
                        for k in range(len(freqi)): freqi[k][j] *= -1
                    freqi = tuple([tuple(f) for f in freqi])
                    FrequenciesToTest_positivenegative.append(freqi)
                    FrequenciesToTest_dic[freqi] = freq
            # 2/2: Permute tuples
            Permutations = list(itertools.permutations(range(m)))
            for freq in FrequenciesToTest_positivenegative:
                for perm in Permutations:
                    freqi = tuple([tuple([freq[i][j] for j in perm]) for i in range(len(freq))])
                    FrequenciesToTest_dic[freqi] = FrequenciesToTest_dic[freq]            
            if verbosity>1: print('There are', len(FrequenciesToTest_dic), 'expanded frequencies to test, out of',len(set(list(FrequenciesToTest_dic.values()))))
            
            # Get OptimalFrequencies
            Proj0 = np.sum([ np.outer(v,v) for v in GramSchmidtOrthonormalizationVectors(frequencies)],0)
            Costs_expanded = dict()
            for freq in FrequenciesToTest_dic:
                Proj = np.sum([ np.outer(v,v) for v in GramSchmidtOrthonormalizationVectors(freq)],0)
                Costs_expanded[freq] = np.linalg.norm(Proj-Proj0)
            Costs_original = {freq:np.inf for freq in FrequenciesToTest}
            FrequenciesToTest_dic_inv = {}
            for freq_new in FrequenciesToTest_dic:
                freq = FrequenciesToTest_dic[freq_new]
                if Costs_expanded[freq_new]<Costs_original[freq]:
                    Costs_original[freq] = Costs_expanded[freq_new]
                    FrequenciesToTest_dic_inv[freq] = freq_new
            Costs = {FrequenciesToTest_dic_inv[freq]:Costs_original[freq] for freq in FrequenciesToTest_dic_inv}            
            OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
                                  
            # Define OptimalLieAlgebra
            OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
            OptimalLieAlgebra = [OptimalChangeOfBasis.T @ A @ OptimalChangeOfBasis for A in OptimalLieAlgebra]

    # Print comments
    if verbosity>0: print('\x1b[1;31m'+'The optimal frequencies found is '+repr(OptimalFrequencies)+'\x1b[0m'+' with cost '+repr(Costs[OptimalFrequencies]))
    if verbosity>1: # print best twelve costs
        scores = [Costs[freq] for freq in Costs]
        indices = np.argsort(scores)
        for i in range(min(len(indices),12)): print('Frequencies', FrequenciesToTest[indices[i]], '- score', round(scores[indices[i]],5), '(best cost #'+repr(i+1)+'/'+repr(len(FrequenciesToTest))+')')
        
    if return_ChangeOfBasis:
        return OptimalFrequencies, OptimalLieAlgebra, OptimalChangeOfBasis
    else:
        return OptimalFrequencies, OptimalLieAlgebra

def OptimizationGrassmann(group, Projection0, frequencies, ambient_dim = 6, dim = 2, determinant = 1, verbosity=0):
    '''
    Given a list of frequencies, build the canonical representative matrices of the torus
    representation, and optimize on special orthogonal matrices O such that the conjugation
    of these matrices yield a subspace of skew-symmetric matrices such that the projection
    on this subspace is close to Projection0.
    
    Input:
        - group: can be 'torus' or 'SO(2)'
        - Projection0: a projection matrix on the space of skew-symmetric matrices
        - frequencies: the frequencies of the representation to test
        - ambient_dim: dimension of the ambient space
        - dim: dimension of the torus, must be equal to len(frequencies)
        - determinant: '+1' or '-1', wheter to optimize over SO(n) or its complement in O(n)
        - verbosity: whether to print comments with pymanopt
        
    Output:
        - result: a pymanopt result
    '''        
    # Define basis
    if group=='torus': CanonicalLieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)
    elif group=='SU(2)': CanonicalLieAlgebra = GetCanonicalLieAlgebraSU2(frequencies)
    
    # Optimization with pymanopt
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def cost(O):
        # Choose connected component in O(n)
        if determinant=='-1': O = O @ np.diag([-1]+[1 for i in range(ambient_dim-1)])
        
        # Conjugate canonical matrices
        LieAlgebra = [O @ CanonicalLieAlgebra[i] @ O.T for i in range(dim)]
        
        # Compute projection
        Projection = FrameToProjection(LieAlgebra)
        
        # Compute distance
        difference = Projection-Projection0
        return np.trace(difference @ difference.T)
    
    # Run optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=verbosity)
    result = optimizer.run(problem)
    
    if determinant=='-1': result.point = result.point @ np.diag([-1]+[1 for i in range(ambient_dim-1)])
    return result

def OptimizationStiefel(group, Sigma, frequencies, ambient_dim = 6, dim = 2, determinant = 1, verbosity=0):
    '''
    Given a list of frequencies, build the canonical representative matrices of the torus
    representation, and optimize on special orthogonal matrices O such that the conjugation
    of these matrices yield a subspace of skew-symmetric matrices such that the projection
    on this subspace is close to Projection0.
    
    Input:
        - group: can be 'torus' or 'SO(2)'
        - Sigma: an operator on the space of nxn matrices
        - frequencies: the frequencies of the representation to test
        - ambient_dim: dimension of the ambient space
        - dim: dimension of the torus, must be equal to len(frequencies)
        - determinant: '+1' or '-1', wheter to optimize over SO(n) or its complement in O(n)
        - verbosity: whether to print comments with pymanopt
        
    Output:
        - result: a pymanopt result
    '''        
    # Define basis
    if group=='torus': CanonicalLieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)
    elif group=='SU(2)': CanonicalLieAlgebra = GetCanonicalLieAlgebraSU2(frequencies)
    CanonicalLieAlgebra = GramSchmidtOrthonormalizationMatrices(CanonicalLieAlgebra)
    
    # Optimization with pymanopt
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def cost(O):
        # Choose connected component in O(n)
        if determinant=='-1': O = O @ np.diag([-1]+[1 for i in range(ambient_dim-1)])

        # Conjugate canonical matrices
        LieAlgebra = CanonicalLieAlgebra
        LieAlgebra = [O @ LieAlgebra[i] @ O.T for i in range(dim)]
        
        # Compute distance
        differences = [Sigma.dot( LieAlgebra[i].flatten() ) for i in range(dim)]
        differences = [np.sum(difference @ difference.T) for difference in differences]
        return np.sum(differences)
    
    # Run optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=verbosity)
    result = optimizer.run(problem)

    if determinant=='-1': result.point = result.point @ np.diag([-1]+[1 for i in range(ambient_dim-1)])
    return result

def GenerateOrbitTorus(LieAlgebra, frequencies, n_points, x, method='random'):
    '''
    method can be 'uniform' or 'random'
    '''
    if method=='uniform':
        # Generate first circle
        coefficient = np.sqrt(2)*np.linalg.norm(frequencies[0])/ComputeGCD(frequencies[0])
        T = np.linspace(0,2*2*np.pi, n_points)
        Orbit = np.array([ scipy.linalg.expm(t*LieAlgebra[0]*coefficient) @ x for t in T])

        # Apply next transformations
        for i in range(1,len(LieAlgebra)):
            coefficient = np.sqrt(2)*np.linalg.norm(frequencies[i])/ComputeGCD(frequencies[i])
            Orbit = [[ scipy.linalg.expm(t*LieAlgebra[i]*coefficient) @ y for t in T] for y in Orbit]
            Orbit = np.concatenate(Orbit)
            
    if method=='random':
        n_points = n_points**len(LieAlgebra)
        TT = np.random.uniform(0,2*2*np.pi, (n_points,len(LieAlgebra)))
        coefficients = [np.sqrt(2)*np.linalg.norm(freq)/ComputeGCD(freq) for freq in frequencies]
        Transformations = []
        for iorbit in range(n_points):
            matrix = np.sum([ coefficients[i]*LieAlgebra[i]*TT[iorbit][i] for i in range(len(LieAlgebra))],0)
            transformation = scipy.linalg.expm(matrix)
            Transformations.append( transformation )
        Orbit = np.array([ transformation @ x for transformation in Transformations ])
        
    return Orbit

def GenerateOrbitSU2(LieAlgebra, frequencies, n_points, x, coefficient=None):
    # Define coefficient and sanity check
    if coefficient is None:
        coefficient = 4*np.pi*np.linalg.norm(GetCanonicalLieAlgebraSU2(frequencies)[0])
    if any([np.linalg.norm(scipy.linalg.expm(coefficient*A)-np.eye(np.shape(A)[0]))>1e-5 for A in LieAlgebra]): 
        print('Error! Incorrect coefficient', coefficient)
        for A in LieAlgebra: print(np.linalg.norm(scipy.linalg.expm(coefficient*A)-np.eye(np.shape(A)[0])))

    # Generate first circle
    T = np.linspace(0,coefficient, n_points)
    Orbit = np.array([ scipy.linalg.expm(t*LieAlgebra[0]) @ x for t in T])
    
    # Apply next transformations
    for i in range(1,3):
        Orbit = [[ scipy.linalg.expm(t*LieAlgebra[i]) @ y for t in T] for y in Orbit]
        Orbit = np.concatenate(Orbit)
        
    return Orbit

'''----------------------------------------------------------------------------
Functions to sample points
----------------------------------------------------------------------------'''

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
        - frequency_max: maximal frequency of the representation
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
    # Define frequencies
#    if frequencies==None: frequencies = random.choice(GetFrequenciesToTest(dim, ambient_dim, frequency_max))
    if frequencies==None: frequencies=GetRandomLattice(dim, ambient_dim, frequency_max)
    if verbose: print('-----> Representation of T^'+repr(dim), 'in R^'+repr(ambient_dim)+'\x1b[1;31m'+' with frequencies '+repr(frequencies)+'\x1b[0m')
        
    # Define a canonical Lie algebra
    LieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)   
        
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
    
    return Sample, LieAlgebra, frequencies

def SampleOnSU2(dim=2, ambient_dim=6, n_points=100, frequencies=None, conjugate=False, right_multiply=False, verbose=False):
    '''
    
    '''        
    # Define frequencies
    FrequenciesToTest = partition_su2(ambient_dim)
    if frequencies==None: 
        FrequenciesToTestAlmostFaithful = [f for f in FrequenciesToTest if len(np.unique(f))==len(f)
                                          and 1 not in f]
        frequencies = random.choice(FrequenciesToTestAlmostFaithful)
    if verbose: print('-----> Representation of SU(2) in R^'+repr(ambient_dim)+'\x1b[1;31m'+' with partition '+repr(frequencies)+'\x1b[0m')
            
    # Define a canonical Lie algebra
    LieAlgebra = GetCanonicalLieAlgebraSU2(frequencies)   
    
    # Get coefficients
    Coefficients = [GetCoefficientSU2(A) for A in LieAlgebra]
    if np.any([np.abs(t-Coefficients[0])>1e-5 for t in Coefficients]): print('Error: the coefficients are not identical')

    coefficient = Coefficients[0]
    if any([np.linalg.norm(scipy.linalg.expm(coefficient*A)-np.eye(np.shape(A)[0]))>1e-5 for A in LieAlgebra]): 
        print('Error! Incorrect coefficient before GramSchmidt', coefficient)
        for A in LieAlgebra: print(scipy.linalg.expm(coefficient*A)) 

    coefficient = Coefficients[0]*np.linalg.norm(LieAlgebra[0])
                
    # Make basis orthonormal    
    LieAlgebra = GramSchmidtOrthonormalizationMatrices(LieAlgebra)
    if any([np.linalg.norm(scipy.linalg.expm(coefficient*A)-np.eye(np.shape(A)[0]))>1e-5 for A in LieAlgebra]): 
        print('Error! Incorrect coefficient', coefficient)
        for A in LieAlgebra: print(scipy.linalg.expm(coefficient*A)) 
        
    # Define orthogonal matrix P and right-multiply
    if right_multiply:
        P = scipy.stats.special_ortho_group.rvs(dim)
        LieAlgebra = [np.sum([LieAlgebra[j]*P[j,i] for j in range(dim)],0) for i in range(dim)]
            
    # Define orthogonal matrix O and conjugate
    if conjugate: O = scipy.stats.special_ortho_group.rvs(ambient_dim)
    else: O = np.eye(ambient_dim)
    for i in range(dim): LieAlgebra[i] = O @ LieAlgebra[i] @ O.T
        
    # Define origin Euclidean vector x
    x = np.ones((ambient_dim,1))
    x /= np.linalg.norm(x)

    # Draw sample from uniform distribution
    dim = 3    
    Sample = []
    Angles = [np.random.uniform(0,coefficient,n_points) for i in range(dim)]
    for i in range(n_points):
        s = x.copy()
        for j in range(dim): 
            s = scipy.linalg.expm(Angles[j][i]*LieAlgebra[j]) @ s
        Sample.append(s[:,0])
    Sample = np.array(Sample)

    return Sample, LieAlgebra, frequencies
