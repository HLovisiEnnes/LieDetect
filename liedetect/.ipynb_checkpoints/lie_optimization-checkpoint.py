from .matrix_manipulation import *
from .lie_pca import *
import autograd.numpy as np 
from autograd import grad   
import pymanopt
import velour
import copy

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

' Optimizations and Lie-PCA'

def GetLiePCAOperator(X,n_neighbors,ambient_dim, dim=None,method='localPCA',verbose=True):
    '''
    method can be localcovariance or localPCA
    '''
    if verbose==True: print('----> Lie PCA with', method, 'estimation of tangent spaces <----')
    
    if method=='localPCA':
        vecs, vals, Sigma = basis_Lie(X, k=n_neighbors, n=dim, plot=False)
#        Sigma = Sigma/len(X)
    
    if method=='localcovariance':
        # Find nice radius
        kdt = sklearn.neighbors.KDTree(X, leaf_size=n_neighbors+1, metric='euclidean')
        NN_Dist, NN = kdt.query(X, n_neighbors+1, return_distance=True)  
        r = np.max(NN_Dist[:,-1])

        # Compute local covariance matrices
        ProjTangentSpaces = velour.NormalizedLocalCovarianceMatrices(X,r)
        ProjTangentSpaces = [ProjTangentSpaces[:,:,i] for i in range(np.shape(X)[0])]

        # Compute normal spaces - robust
        ProjTangentSpaces = [(proj+proj.T)/2 for proj in ProjTangentSpaces] #symmetrize
        ProjTangentSpaces = [proj/np.linalg.norm(proj)*np.sqrt(dim) for proj in ProjTangentSpaces] #normalize
        ProjNormalSpaces = [np.eye((ambient_dim))-proj for proj in ProjTangentSpaces] #complementary
        
#         # Compute normal spaces - eigencut
#         ProjNormalSpaces = []
#         for i in range(np.shape(X)[0]):
#             proj = ProjTangentSpaces[i]
#             eigenvalues , Z = scipy.linalg.eigh(proj) 
#             Tproj = np.eye(ambient_dim)
#             for j in range(dim):
#                 index = np.argsort(eigenvalues)[-(j+1)] #the ith largest eigenvalue
#                 Tproj[index,index] = 0
#             ProjNormalSpaces.append( Z @ Tproj @ Z.T )        

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

#         # Normalize
#         Sigma = Sigma/np.linalg.norm(Sigma)

    # Get eigenvectors
    vals, vecs = np.linalg.eig(Sigma) #finds eigenvalues and vectors of sigma as a matrix
    vals = np.real(vals)
    indices = np.argsort(vals)
    vals, vecs = [vals[i] for i in indices], list([vecs[:,i] for i in indices])

    if verbose==True: print('First eigenvalues:', vals[0:4])
#    if verbose==True: print('Ratio between first eigenvalues:', vals[1]/vals[0])
        
    return Sigma

def FindClosestLieAlgebra(Sigma, dim, ambient_dim, frequency_max=None, FrequenciesToTest=None, method='Grassmann',verbosity=1):
    '''
    Input: 
    - method can be Grassmann or Stiefel or NormalForm
    - verbosity: 0 (no verbose), 1 (a few comments), 2 (comments from pymanopt)
    '''
    if verbosity>0: print('----> Optimization problem via', method, '<----')

    if method == 'Grassmann':
        # Get FrequenciesToTest
        if FrequenciesToTest==None:
            FrequenciesToTest = GetFrequenciesToTest(dim=dim, ambient_dim=ambient_dim, frequency_max=frequency_max)
        if verbosity>1: print('There are', len(FrequenciesToTest), 'frequencies to test.')

        # Get and sort eigenvectors of Sigma
        vals, vecs = np.linalg.eig(Sigma) # finds eigenvalues and vectors of sigma as a matrix
        vecs = [vecs[:,i] for i in range(len(vecs))]
        indices = np.argsort(vals)        # argsort eigenvalues
        vals, vecs = [vals[i] for i in indices], [vecs[i] for i in indices]

        # Get bottom Eigenvector of Sigma
        LieAlgebra = [vecs[i].reshape((ambient_dim,ambient_dim)) for i in range(dim)]
        LieAlgebra = [(A-A.T)/2 for A in LieAlgebra]
        
        # Define Projection0, the projection matrix on the vector subspace given by Lie PCA
        Projection0 = FrameToProjection(LieAlgebra)

        # Run optimization with pymanopt
        Costs = dict() 
        ChangeOfBasis = dict()
        for frequencies in FrequenciesToTest:    
            result = OptimizationTorusGrassmann(Projection0, frequencies, ambient_dim, dim, verbosity=0)    
            Costs[frequencies], ChangeOfBasis[frequencies] = result.cost, result.point
            if verbosity>0: print('Frequencies', frequencies, '- result', result.cost)
            if verbosity>1: print('('+result.stopping_criterion+')')

        # Define optimal angle and cost
        OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
        O = ChangeOfBasis[OptimalFrequencies]
        if verbosity>0: print('The optimal frequencies found is', OptimalFrequencies, 'with cost', min(Costs.values()))

        # Define OptimalLieAlgebra
        OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
        OptimalLieAlgebra = [O @ OptimalLieAlgebra[i] @ O.T for i in range(dim)]
        OptimalLieAlgebra = GramSchmidtOrthonormalizationMatrices(OptimalLieAlgebra)
    
    if method == 'Stiefel':
        # Get FrequenciesToTest
        if FrequenciesToTest==None:
            FrequenciesToTest = GetFrequenciesToTest(dim=dim, ambient_dim=ambient_dim, frequency_max=frequency_max)
        if verbosity>1: print('There are', len(FrequenciesToTest), 'frequencies to test.')

        # Run optimization with pymanopt
        Costs = dict() 
        ChangeOfBasis = dict()
        for frequencies in FrequenciesToTest:    
            result = OptimizationTorusStiefel(Sigma, frequencies, ambient_dim, dim, verbosity=0)    
            Costs[frequencies], ChangeOfBasis[frequencies] = result.cost, result.point
            if verbosity>0: print('Frequencies', frequencies, '- result', result.cost)
            if verbosity>1: print('('+result.stopping_criterion+')')

        # Define optimal angle and cost
        OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
        O, P = ChangeOfBasis[OptimalFrequencies]
        if verbosity>0: print('The optimal frequencies found is ', OptimalFrequencies, 'with cost ', min(Costs.values()))

        # Define OptimalLieAlgebra
        OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
        OptimalLieAlgebra = GramSchmidtOrthonormalizationMatrices(OptimalLieAlgebra)
        if dim>1: OptimalLieAlgebra = (np.array([A.flatten() for A in OptimalLieAlgebra]).T @ P).T.reshape((dim, ambient_dim, ambient_dim))
            #Right-multiplication canonical matrices
        OptimalLieAlgebra = [O @ OptimalLieAlgebra[i] @ O.T for i in range(dim)]
            # Conjugate canonical matrices

    if method == 'NormalForm':
        # Get and sort eigenvectors of Sigma
        vals, vecs = np.linalg.eig(Sigma) # finds eigenvalues and vectors of sigma as a matrix
        vecs = [vecs[:,i] for i in range(len(vecs))]
        indices = np.argsort(vals)        # argsort eigenvalues
        vals, vecs = [vals[i] for i in indices], [vecs[i] for i in indices]

        # Get bottom Eigenvector of Sigma
        LieAlgebra = [vecs[i].reshape((ambient_dim,ambient_dim)) for i in range(dim)]
        LieAlgebra = [(A-A.T)/2 for A in LieAlgebra]

        if dim==1:    
            # Estimate frequencies
            angles, Z, T = NormalFormSkewSymmetricMatrix(LieAlgebra[0])
            angles = np.sort(np.abs(angles))
            if verbosity>0: print('Frequencies in normal form decomposition:', tuple(angles))

            # Generate frequencies to test
            FrequenciesToTest = GetFrequenciesToTest(dim=1, ambient_dim=ambient_dim, frequency_max=frequency_max)
            FrequenciesToTestNorm = [list(f[0])/np.linalg.norm(f[0]) for f in FrequenciesToTest]

            # Find closest frequencies
            i = np.argmin([np.linalg.norm(f-angles) for f in FrequenciesToTestNorm])
            OptimalFrequencies = FrequenciesToTest[i]
            if verbosity>0: print('The optimal frequencies found is ', OptimalFrequencies, 'with cost ', np.linalg.norm(FrequenciesToTestNorm[i]-angles))

            # Define Lie algebra
            L = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
            OptimalLieAlgebra = [Z @ L[0]/np.linalg.norm(L[0]) @ Z.T]

        else: #dim >1
            # Get and sort eigenvectors of Sigma
            vals, vecs = np.linalg.eig(Sigma) # finds eigenvalues and vectors of sigma as a matrix
            vecs = [vecs[:,i] for i in range(len(vecs))]
            indices = np.argsort(vals)        # argsort eigenvalues
            vals, vecs = [vals[i] for i in indices], [vecs[i] for i in indices]

            # Get bottom Eigenvector of Sigma
            LieAlgebra = [vecs[i].reshape((ambient_dim,ambient_dim)) for i in range(dim)]
            LieAlgebra = [(A-A.T)/2 for A in LieAlgebra]
            
            # Orthonormalize 
            LieAlgebra = GramSchmidtOrthonormalizationMatrices(LieAlgebra)
            if verbosity>0: print('Test commutation:', np.linalg.norm(LieAlgebra[0] @ LieAlgebra[1] - LieAlgebra[1] @ LieAlgebra[0]))

            # Get frequencies via normal form
            frequencies, O = NormalFormSkewSymmetricMatrices(LieAlgebra)
            
            # Get FrequenciesToTest
            FrequenciesToTest0 = GetFrequenciesToTest(dim=dim, ambient_dim=ambient_dim, frequency_max=frequency_max)
            m = int(ambient_dim/2)
            # Expand GetFrequenciesToTest by permutation and negative values
            FrequenciesToTest1 = []
            FrequenciesToTest_dic = {} #dict to remember the initial value of freq
            # Put negative values
            for freq in FrequenciesToTest0:
                freq = [list(f) for f in freq]
                indices = powerset(range(m))
                for i in range(len(indices)):
                    freqi = copy.deepcopy(freq)
                    ind = indices[i]
                    for j in ind:
                        for k in range(len(freqi)):
                            freqi[k][j] *= -1
                    freqi = tuple([tuple(f) for f in freqi])
                    FrequenciesToTest1.append(freqi)
                    FrequenciesToTest_dic[freqi] = freq
            # Permute tuples
            Permutations = list(itertools.permutations(range(m)))
            FrequenciesToTest = []
            for freq in FrequenciesToTest1:
                for perm in Permutations:
                    freqi = tuple([tuple([freq[i][j] for j in perm]) for i in range(len(freq))])
                    FrequenciesToTest.append(freqi)
                    FrequenciesToTest_dic[freqi] = FrequenciesToTest_dic[freq]            
            if verbosity>1: print('There are', len(FrequenciesToTest), 'frequencies to test.')

            # Get OptimalFrequencies
            Proj0 = np.sum([ np.outer(v,v) for v in GramSchmidtOrthonormalizationVectors(frequencies)],0)
            Costs = dict()
            for freq in FrequenciesToTest:
                Proj = np.sum([ np.outer(v,v) for v in GramSchmidtOrthonormalizationVectors(freq)],0)
                Costs[freq] = np.linalg.norm(Proj-Proj0)
            OptimalFrequencies = min(Costs.keys(), key=(lambda k: Costs[k]))
            if verbosity>0: 
                print('The optimal frequencies found is', OptimalFrequencies, 'cost:', Costs[OptimalFrequencies])
                OptimalFrequenciesCanonical = tuple([tuple(f) for f in FrequenciesToTest_dic[OptimalFrequencies]])
                print('              ---> equivalent to', OptimalFrequenciesCanonical)
                
            # Define OptimalLieAlgebra
            OptimalLieAlgebra = GetCanonicalLieAlgebraTorus(OptimalFrequencies)
            OptimalLieAlgebra = [O.T @ A @ O for A in OptimalLieAlgebra]
            
    return OptimalFrequencies, OptimalLieAlgebra

def OptimizationTorusGrassmann(Projection0, frequencies, ambient_dim = 6, dim = 2, verbosity=0):
    '''
    Given a list of frequencies, build the canonical representative matrices of the torus
    representation, and optimize on special orthogonal matrices O such that the conjugation
    of these matrices yield a subspace of skew-symmetric matrices such that the projection
    on this subspace is close to Projection0.
    
    Input:
        - Projection0: a projection matrix on the space of skew-symmetric matrices
        - frequencies: the frequencies of the representation to test
        - ambient_dim: dimension of the ambient space
        - dim: dimension of the torus, must be equal to len(frequencies)
        - verbosity: whether to print comments with pymanopt
        
    Output:
        - result: a pymanopt result
    '''        
    # Define basis
    CanonicalLieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)
    
    # Optimization with pymanopt
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def cost(O):
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

    return result

def OptimizationTorusStiefel(Sigma, frequencies, ambient_dim = 6, dim = 2, verbosity=0):
    '''
    Given a list of frequencies, build the canonical representative matrices of the torus
    representation, and optimize on special orthogonal matrices O such that the conjugation
    of these matrices yield a subspace of skew-symmetric matrices such that the projection
    on this subspace is close to Projection0.
    
    Input:
        - Sigma: an operator on the space of nxn matrices
        - frequencies: the frequencies of the representation to test
        - ambient_dim: dimension of the ambient space
        - dim: dimension of the torus, must be equal to len(frequencies)
        - verbosity: whether to print comments with pymanopt
        
    Output:
        - result: a pymanopt result
    '''        
    # Define basis
    CanonicalLieAlgebra = GetCanonicalLieAlgebraTorus(frequencies)
    CanonicalLieAlgebra = GramSchmidtOrthonormalizationMatrices(CanonicalLieAlgebra)
    
    # Optimization with pymanopt
    manifold_O = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)
    manifold_P = pymanopt.manifolds.SpecialOrthogonalGroup(dim, k=1)
    manifold = pymanopt.manifolds.product.Product([manifold_O,manifold_P])

    @pymanopt.function.autograd(manifold)
    def cost(O,P):
        # Right-multiplication canonical matrices
        if dim>1: LieAlgebra = (np.array([A.flatten() for A in CanonicalLieAlgebra]).T @ P).T.reshape((dim, ambient_dim, ambient_dim))
        else: LieAlgebra = CanonicalLieAlgebra        

        # Conjugate canonical matrices
        LieAlgebra = [O @ LieAlgebra[i] @ O.T for i in range(dim)]
        
        # Compute distance
        differences = [Sigma.dot( LieAlgebra[i].flatten() ) for i in range(dim)]
        differences = [np.sum(difference @ difference.T) for difference in differences]
        return np.sum(differences)
    
    # Run optimization
    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(verbosity=verbosity)
    result = optimizer.run(problem)

    return result

def GenerateOrbitTorus(LieAlgebra, frequencies, n_points, x, dim):
    T = np.linspace(0,2*2*np.pi, n_points)
    
    # Generate first circle
    coefficient = np.sqrt(2)*np.linalg.norm(frequencies[0])/ComputeGCD(frequencies[0])
    Orbit = np.array([ scipy.linalg.expm(t*LieAlgebra[0]*coefficient) @ x for t in T])
    
    # Apply next transformations
    for i in range(1,dim):
        coefficient = np.sqrt(2)*np.linalg.norm(frequencies[i])/ComputeGCD(frequencies[i])
        Orbit = [[ scipy.linalg.expm(t*LieAlgebra[i]*coefficient) @ y for t in T] for y in Orbit]
        Orbit = np.concatenate(Orbit)

    return Orbit
