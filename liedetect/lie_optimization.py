from .matrix_manipulation import *
from .lie_pca import *
import autograd.numpy as np 
from autograd import grad   
import pymanopt
import velour
import copy
import skdim

# Misc
import time, datetime
import random
import itertools
import sys
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
    
    

' Lie Algebra manipulations '

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



def AreFrequencesEquivalent(freq0,freq1,verbose=True):    
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
    Irreps = list(itertools.product(range(-frequency_max,frequency_max+1), repeat=dim))
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


def GetPrimitiveLattices(dim, ambient_dim, frequency_max, verbose=0):
    m = int(ambient_dim/2)

    # Initial list of frequencies: Cartesian product
    if verbose>0: start_time = ChronometerStart('Decomposition into irreps... ')
    Irreps = list(itertools.product(range(-frequency_max,frequency_max+1), repeat=dim))
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
    if verbose>0: print('Number of frequences to test:', len(FrequenciesToTest))
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
        FrequenciesToTest = GetPrimitiveLattices(dim, ambient_dim, frequency_max, verbose=verbose)
    if verbose>-1: print('Number of frequencies to test:', len(FrequenciesToTest))
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

def GetLiePCAOperator(X,n_neighbors,dim=None,method='localPCA',verbose=True):
    '''
    method can be localcovariance or localPCA
    '''
    if verbose==True: print('----> Lie PCA with', method, 'estimation of tangent spaces <----')
    ambient_dim = np.shape(X)[1]
    
    # Find nice radius
    kdt = sklearn.neighbors.KDTree(X, leaf_size=n_neighbors+1, metric='euclidean')
    NN_Dist, NN = kdt.query(X, n_neighbors+1, return_distance=True)  
    r = np.max(NN_Dist[:,-1])

    # Compute local covariance matrices
    ProjTangentSpaces = velour.NormalizedLocalCovarianceMatrices(X,r)
    ProjTangentSpaces = [ProjTangentSpaces[:,:,i] for i in range(np.shape(X)[0])]
    ProjTangentSpaces = [(proj+proj.T)/2 for proj in ProjTangentSpaces] #symmetrize
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

    # Get eigenvectors
    vals, vecs = np.linalg.eig(Sigma) #finds eigenvalues and vectors of sigma as a matrix
    vals = np.real(vals)
    indices = np.argsort(vals)
    vals, vecs = [vals[i] for i in indices], list([vecs[:,i] for i in indices])

    if verbose==True: print('First eigenvalues:', vals[0:4])
        
    return Sigma

def FindClosestLieAlgebra(group, Sigma, dim=None, ambient_dim=4, frequency_max=3, FrequenciesToTest=None, method='Grassmann',verbosity=1):
    '''
    Input: 
    - group: can be 'torus', 'SU(2)' or 'SO(3)'
    - dim: dimension of the torus, if group='torus'
    - method: can be 'Grassmann' or 'Stiefel' or 'NormalForm' (the latter only for group='torus')
    - verbosity: 0 (no verbose), 1 (a few comments), 2 (comments from pymanopt)
    '''
    if verbosity>0: print('----> Optimization problem via'+'\x1b[1;31m', method, '\x1b[0m'+'for', group, 'group <----')

    # Get FrequenciesToTest
    if FrequenciesToTest==None:
        if group=='torus': FrequenciesToTest = GetFrequenciesToTest(dim=dim, ambient_dim=ambient_dim, frequency_max=frequency_max)
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
        O = ChangeOfBasis[OptimalFrequencies]
            
        # Define OptimalLieAlgebra
        if group=='torus': OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
        if group=='SU(2)': OptimalLieAlgebra = GetCanonicalLieAlgebraSU2(OptimalFrequencies)
        OptimalLieAlgebra = [O @ OptimalLieAlgebra[i] @ O.T for i in range(dim)]
        OptimalLieAlgebra = GramSchmidtOrthonormalizationMatrices(OptimalLieAlgebra)
    
    elif method in ['NormalForm']:
        if dim==1:    
            # Estimate frequencies
            angles, Z, T = NormalFormSkewSymmetricMatrix(LieAlgebra[0])
            angles = np.sort(np.abs(angles))
            if verbosity>1: print('Normalized frequencies found in normal form decomposition:', tuple(angles))

            # Generate frequencies to test
            FrequenciesToTestNorm = [list(f[0])/np.linalg.norm(f[0]) for f in FrequenciesToTest]

            # Find closest frequencies
            Costs = {FrequenciesToTest[i]:np.linalg.norm(FrequenciesToTestNorm[i]-angles) for i in range(len(FrequenciesToTest))} 
            OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
                                                    
            # Define Lie algebra
            L = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
            OptimalLieAlgebra = [Z @ L[0]/np.linalg.norm(L[0]) @ Z.T]
            
        else: #dim >1
            # Orthonormalize 
            LieAlgebra = GramSchmidtOrthonormalizationMatrices(LieAlgebra)
            if verbosity>1: print('Test commutation:', np.linalg.norm(LieAlgebra[0] @ LieAlgebra[1] - LieAlgebra[1] @ LieAlgebra[0]))

            # Get frequencies via normal form
            frequencies, O = NormalFormSkewSymmetricMatrices(LieAlgebra)
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
            OptimalLieAlgebra = [O.T @ A @ O for A in OptimalLieAlgebra]

    # Print comments
    if verbosity>0: print('\x1b[1;31m'+'The optimal frequencies found is '+repr(OptimalFrequencies)+'\x1b[0m'+' with cost '+repr(Costs[OptimalFrequencies]))
    if verbosity>1: # print best twelve costs
        scores = [Costs[freq] for freq in Costs]
        indices = np.argsort(scores)
        for i in range(min(len(indices),12)): print('Frequencies', FrequenciesToTest[indices[i]], '- score', round(scores[indices[i]],5), '(best cost #'+repr(i+1)+'/'+repr(len(FrequenciesToTest))+')')
            
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
        for i in range(1,dim):
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

def GenerateOrbitSU2(LieAlgebra, frequencies, n_points, x):
    # Define coefficient and sanity check
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