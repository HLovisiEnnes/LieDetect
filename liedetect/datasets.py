import autograd.numpy as np 
import scipy, sklearn
import velour
import matplotlib.pyplot as plt
import random
import os
import shutil
import random
from .matrix_manipulation import *
from .lie_optimization import *

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

def GorillaImages(dim = 1, file = "./RotatingImages/gorilla_white.png", pas = 7, save = None, verbosity = 1):
    '''
    Generates a data set of images of a gorilla translated on the y-direction or x and y directions (see ReadMe). 
    Input:
        - dim: if 1, the gorilla is translated on the y-direction only; if 2 it is translated on the x and y directions
        - file: path to where to find the initial gorilla
        - pas: parameter to decrease the image sizes
        - save : path to save the numpy array of output, if None, does not save
        - vebosity: if 2 full, otherwise gives only output
    Output:
        - X: array corresponding to data set of trasnlated images
    '''
    # Open file
    pic = plt.imread(file)
    pic = pic[::pas,::pas,0:3]
    pic = pic[7:137,9:130]

    shape = np.shape(pic)
    x = np.shape(pic)[0]
    y = np.shape(pic)[1]

    # Translate image
    X = []
    x2, y2 = int(x/2), int(y/2)
    
    if dim == 1:
        X = np.zeros((x2,x*y*3))
        for i in range(x2):
            pic2 = np.roll(pic, 2*i, 0)
            X[i] = pic2.flatten()
        shape = tuple([130, 120, 3])
        x, y = shape[0], shape[1]
        N1 = 65
        N = N1 
        len_img = 130*120*3
        
    elif dim == 2:
        X = np.zeros((x2,y2,x*y*3))
        for i in range(x2):
            for j in range(y2):
                pic2 = np.roll(pic, 2*i, 0)
                pic2 = np.roll(pic2, 2*j, 1)
                X[i,j] = pic2.flatten()
            
        shape = tuple([130, 120, 3])
        x = shape[0]; y = shape[1]
        N1 = 65; N2 = 60
        N = N1*N2
        len_img = 130*120*3
    else:
        raise Exception("Invalid parameter value: dim must be either 1 or 2!")

    #saves array if desired
    if save != None:
        np.save('RotatingImages/' + save, X)

    if verbosity == 2:
        if dim == 1:
            print('shape X:', np.shape(X))
            # Plot some images of the dataset
            for i in np.linspace(0, N1, 3)[:-1]:
                i = int(i)
                plt.figure(figsize=(3,3))
                plt.imshow(X[i].reshape(shape))
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.gca().axes.get_yaxis().set_visible(False)    
        else:
            print('shape X:', np.shape(X))
            # Plot some images of the dataset
            for j in list(np.linspace(0, N2, 4)[:-1])+[N2-2]:
                for i in np.linspace(0, N1, 3)[:-1]:
                    j = int(j); i = int(i)
                    plt.figure(figsize=(3,3))
                    plt.imshow(X[i,j].reshape(shape))
                    plt.gca().axes.get_xaxis().set_visible(False)
                    plt.gca().axes.get_yaxis().set_visible(False)
    X = X.reshape((N, len_img))
    X = X/max([np.linalg.norm(X[i]) for i in range(np.shape(X)[0])])
    return X

def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]) #transforms rgb into gray scale


def ArrowImages(N_images = 200, verbosity = 1, save = False):
    '''
    Makes a data set of arrows rotated about their centers
    Input:
        - N_images: the number of points in the data set
        - vebosity: if 2 full, otherwise gives only output
        - save: if False, deletes arrow images, otherwise saves in RotatingImages/Arrows
     Output:
        - X: array corresponding to data set of trasnlated images
    '''
    if os.path.isdir('RotatingImages/Arrows') == False: os.mkdir('RotatingImages/Arrows') #makes folder to save images
    for value in range(0, N_images):
        fig, ax = plt.subplots()
        ax.arrow(0.5,0.5,0.3*np.cos(np.pi*value/100),0.3*np.sin(np.pi*value/100), width = 0.1, length_includes_head = True, 
                 head_width = 0.3, color = 'black') #make the arrows centered at 0.5x0.5 and rotate them in a full turn in intervals of 360/20 degrees
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.gca().set_aspect('equal', adjustable='box') #make scale of axes as similar as possible
        plt.axis('off')  
        plt.savefig('RotatingImages/Arrows/'+'rot'+str(value)+'.png', dpi=50) #dpi controls the number of pixels used
        plt.close() 

    if verbosity == 2:
        # Plot an image
        image = plt.imread('RotatingImages/Arrows/'+'rot10.png')
        gray = rgb2gray(image)    
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.axis('off')
        plt.show()
        np.shape(gray)
    
    rot_dataset = []
    for value in range(0,N_images):
        image = plt.imread('RotatingImages/Arrows/'+'rot'+str(value)+'.png')
        gray = rgb2gray(image)    
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.axis('off')
        np.shape(gray)
        rot_dataset.append(gray.reshape(-1))
        plt.close() 

    X = rot_dataset
    X = np.array(X)
    for i in range(np.shape(X)[0]): X[i,:] /= np.linalg.norm(X[i,:])
    
    if save == False:
        shutil.rmtree('/RotatingImages/Arrows')
    return X

def rotate_image(image, angle):
    '''
    Roatetes image by angle degrees around its center
    Input:
        - image: rank 2 tensor to be rotated
        - angle: value in degrees to rotate the image
    Output:
        - output: rank 2 tensor of rotated image
    Code from https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    OBS.: Requires cv2.''' 
    import cv2
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def RotMNIST(number_of_rotations = 50, size_train = 1000, size_test = 50):
    '''
    Generates a data set of size_train MNIST images and number_of_rotations random rotations about their centers, together with size_test randomly rotated images about their center
    Input:
        - number_of_rotations: number of random rotations to each train image
        - size_train: total number of images to be rotated
        - size_test: total test size
    Output: X_tain, y_train, X_test, y_test
    Notice that the shapes are
        - X_train: (number_of_rotations*size_train,28,28)
        - y_train: (number_of_rotations*size_train,)
        - X_test: (size_test,28,28)
        - y_test: (size_test,)
    OBS.: Requires keras and cv2. 
    '''
    from keras.datasets import mnist
    
    #downloads MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    #takes random sets from train and test
    subset_train = random.sample(range(len(X_train)), size_train)
    subset_test = random.sample(range(len(X_test)), size_test)
    
    #makes train and test sets
    X_train, y_train = X_train[subset_train], y_train[subset_train]
    X_test, y_test = X_test[subset_test], y_test[subset_test]
    
    #makes data set of rotated images for both training and test
    X_train_rot = []
    y_train_rot = []
    
    for im_index, im in enumerate(X_train):
        for i in range(number_of_rotations):
            X_train_rot.append(rotate_image(im, random.uniform(0, 360)))
            y_train_rot.append(y_train[im_index])
    
    X_test_rot = []
    y_test_rot = []
    
    for im_index, im in enumerate(X_test):
        X_test_rot.append(rotate_image(im, random.uniform(0, 360)))
        y_test_rot.append(y_test[im_index])
            
    return np.array(X_train_rot), np.array(y_train_rot), np.array(X_test_rot), np.array(y_test_rot)

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

def SampleOnSU2(dim=2, ambient_dim=6, n_points=100, frequencies=None, conjugate=True, right_multiply=True, verbose=False):
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
    if verbose: print('Coefficient:', round(coefficient/(4*np.pi),4))
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