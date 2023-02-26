import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from sklearn.decomposition import PCA

def local_PCA(data, k, n):
    '''
    Estimates the tangent spaces to each point, T_i, by local PCA, that is, applying PCA to the k closest
    points to each point
    Input: data: transformed dataset; k: number of closest points to be considered in nearest neighbors;
    n: number of components to consider for PCA
    Output: estimates T_i
    '''
    Ts = []
    mat = [[0]*(len(data)) for el in range(len(data))] #makes a zero len(data) x len(data) matrix for iteration

    #we calculate the symmetric len(data) x len(data) distance matrix for our data
    for i in range(len(data)):
        for j in range(i):
            dist = np.linalg.norm(data[i]-data[j])
            mat[i][j] = dist
            mat[j][i] = dist 

            
    #for each point, pick the k closest points and apply PCA
    for i in range(len(data)):
        dis = mat[i] #list of distances of all points from point i
        indices = list(range(len(data))) #list of points indices
        
        list1, list2 = zip(*sorted(zip(dis, indices))) #sort the list of indiced based on the list of distances
        
        closest_indices = list2[1 : k+1] #pick the k closest points to i, ignoring i itself
        
        data_for_pca = [data[i]]
        #makes dataset for local PCA
        for j in closest_indices:
            data_for_pca.append(data[j])
        
        #apply PCA to local data
        pca = PCA(n_components = n)
        pca.fit(data_for_pca)
        Ts.append(pca.components_)
        
    return Ts

def vis_Ti(data, Ts, i, deltaT, save = None):
    '''
    Allows for visualizing the estimated tangent spaces for data in \R^2. It plots the full data set, together
    with the specified ith point, x_i, in red and the associated estimated tangent space T_i as a red line segment
    given by x_i + deltaT T_i
    Input: data; Ts: estimations of tangent spaces for each data point; deltaT: how big the lines representing
    the tangent spaces are; save = None: if not None, saves the visualization as "save"
    Output: None
    '''
    delta_coord = Ts[i][0] #take the vector in Ts for the point i
    point = data[i] #picks the ith point
    
    coord_i = -deltaT/2*delta_coord + point #makes initial coordinates of the line
    coord_f = deltaT/2*delta_coord + point #makes final coordinates of the line
    
    if save != None:
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(point[0], point[1], c = 'r')
        plt.plot([coord_i[0], coord_f[0]], [coord_i[1], coord_f[1]], c = 'r')
        plt.axis('equal')
        plt.savefig(save)
        plt.show()
    else:
        plt.scatter(data[:,0], data[:,1])
        plt.scatter(point[0], point[1], c = 'r')
        plt.plot([coord_i[0], coord_f[0]], [coord_i[1], coord_f[1]], c = 'r')
        plt.axis('equal')
        plt.savefig(save)
        plt.show()
        
    return None

def perp_Ts(Ts):
    '''
    Estimates T_i^\perp by null space matrices applied to T_i for each T_i
    Input: Ts: T_i
    Output: estimates of T_i^\perp
    '''
    Ts_perpendicular = []
    for element in Ts:
        Ts_perpendicular.append(null_space(element).transpose())
    return Ts_perpendicular


def basis_Lie(data, Ts_perp = None, dim = None, k = None, n = None, plot = True, save = None):
    '''
    We will compute the matrix \Sigma acting on \R^{n\times n} as \Sigma = \Sigma([b_1,...,b_{n^2}]), as 
    \{b_1,...,b_{n^2}\} is the usual basis of \R^{n\times n}
    Input: data, Ts_perp = None: if None, estimates T_i^\perp by local PCA for a given k and n, else inputs
    estimations of T_i^\perp (it should be a list or array for which each points T_i^\perp is a matrix, where
    the rows are basis elements for this space); dim = None: if None, is set to data.shape[1], else same as dim(data)
    k: number of closest points to be considered in nearest neighbors (only if Ts_perp == None); 
    n: number of components to consider for PCA (only if Ts_perp == None);
    plot = True: if True, plots the eigenvalue plot, which works just as explained variance ratio;
    save = None: if save != None, saves the eigenvalue plot, which works just as explained variance ratio
    Output: (vecs, val) estimated vectors for a basis of the Lie algebra, in order, and the associated eigenvalues
    '''        
    if dim == None:
        dim = data.shape[1]
        
    if Ts_perp == None:
        Ts = local_PCA(data, k, n)
        Ts_perp = perp_Ts(Ts)
     
    sigma = np.zeros((dim**2, dim**2))
    
    for i in range(dim**2):
        #makes the basis elements for R^{dim \times dim} seen as a matrix space
        basis = [0 if el != i else 1 for el in range(dim**2)] 
        basis = np.array(basis)
        basis = basis.reshape(dim, dim)
                
        vec = np.zeros(dim**2)
        for j, data_point in enumerate(data):
            
            proj_T = np.zeros((dim, dim))
            for k in range(Ts_perp[j].shape[0]):
                proj_T += np.outer(Ts_perp[j][k], Ts_perp[j][k]) #proj_T^\perp
                
            proj_span = np.outer(data_point, data_point)*1/(np.dot(data_point, data_point)) #\proj_span{x_i}
            
            vec_mat = proj_T @ basis @ proj_span
            vec += vec_mat.reshape(1,-1)[0]
            
        sigma[:,i] = vec
        
    vals, vecs = np.linalg.eig(sigma) #finds eigenvalues and vectors of sigma as a matrix
    vec = [] 
    
    for i in range(vecs.shape[1]): #separate the eigenvectors of the solution in a list
        vec.append(vecs[:,i])
        
    vals_list = vals.tolist()
    vals, vecs = zip(*sorted(zip(vals_list, vec))) #order the eigenvectors based on eigenvalues
    
    #makes the plot of eigenvalues, which works just as explained variance ratio
    if plot == True:
        my_xticks = []
        for i in range(1, len(vals) +1):
            my_xticks.append('$\\lambda_' + str(i)+ '$')

        if save != None:
            plt.xticks(range(len(vals)), my_xticks)
            plt.plot(range(len(vals)), vals, 'o--')
            plt.savefig(save)
            plt.show()
        else:
            plt.xticks(range(len(vals)), my_xticks)
            plt.plot(range(len(vals)), vals, 'o--')
            plt.show()
    
    return vecs, vals