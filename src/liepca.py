"""---------------------------------------------------------------------------------------------------------------------

LieDetect: Detection of representation orbits of compact Lie groups from point clouds
Henrique Ennes & Raphaël Tinarrage
See the repo at https://github.com/HLovisiEnnes/LieDetect and the article at https://arxiv.org/abs/2309.03086

------------------------------------------------------------------------------------------------------------------------

This module provides functions for dimension reduction and orthonormalization (Step 1 of LieDetect), as well as Lie PCA
(Step 2).

------------------------------------------------------------------------------------------------------------------------

Dimension reduction:
    get_covariance_matrix
    print_covariance_eigenvalues
    project_on_minimal_subspace

Orthonormalization:
    print_norms
    orthonormalize

Lie PCA:
    get_lie_pca_operator

---------------------------------------------------------------------------------------------------------------------"""

# Standard imports.
from typing import Optional

# Third-party imports.
# import autograd.numpy as np
import numpy as np
import scipy, sklearn


"""---------------------------------------------------------------------------------------------------------------------
Dimension reduction
---------------------------------------------------------------------------------------------------------------------"""


def get_covariance_matrix(
    pts: np.ndarray,
    center: bool = False,
    normalize: bool = False,
    orbit_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Computes the covariance matrix of a point cloud.

    Args:
        pts (np.ndarray): Array representing the points.
        center (bool, optional): If True, centers the points before computing the covariance. Defaults to False.
        normalize (bool, optional): If True, normalizes the covariance matrix by its Frobenius norm and rescales by
            sqrt(orbit_dim). This normalization makes it close to a projection matrix. Defaults to False.
        orbit_dim (int, optional): Dimension used for normalization. Required if normalize is True.

    Returns:
        np.ndarray: The covariance matrix of the points.
    """
    # Center if needed.
    if center:
        pts = pts - np.mean(pts, axis=0)
    # Compute covariance matrix.
    cov = np.mean([np.outer(pt, pt) for pt in pts], axis=0)
    # Normalize if needed.
    if normalize:
        cov = cov / np.linalg.norm(cov) * np.sqrt(orbit_dim)
    return cov


def print_covariance_eigenvalues(pts) -> None:
    """Prints the eigenvalues of the covariance matrix of the given points in decreasing order."""
    # Compute covariance matrix.
    cov = get_covariance_matrix(pts)
    # Compute eigenvalues and normalize.
    eigenvalues = np.sort(np.linalg.eigvals(cov).real)[::-1]
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    # Print result.
    print("Covariance eigenvalues:", *[f"{v:.1e} " for v in eigenvalues])


def project_on_minimal_subspace(
    pts: np.ndarray, threshold_eigenvalue: float
) -> np.ndarray:
    """
    Projects the points onto the minimal subspace they span, based on their covariance matrix. The projection is done by
    removing components with eigenvalues below a certain threshold, after normalization by L1 norm of eigenvalues.

    Args:
        pts (np.ndarray): Array of shape (nb_points, ambient_dim) representing the sampled points.
        threshold_eigenvalue (float, optional): Threshold for eigenvalues to consider a component significant.

    Returns:
        np.ndarray: Projected points in the minimal subspace.
    """
    # Compute covariance matrix.
    cov = get_covariance_matrix(pts)
    # Get eigenvalues and eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Normalize eigenvalues by L1 norm.
    eigenvalues = eigenvalues / np.sum(eigenvalues)
    # Filter out small eigenvalues.
    significant_indices = np.where(eigenvalues > threshold_eigenvalue)[0]
    # Project onto the minimal subspace.
    projected_pts = pts @ eigenvectors[:, significant_indices]
    return projected_pts


"""---------------------------------------------------------------------------------------------------------------------
Orthonormalization
---------------------------------------------------------------------------------------------------------------------"""


def print_norms(pts: np.ndarray) -> None:
    # Print the mean and standard deviation of the norms of the points.
    norms = np.linalg.norm(pts, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    print(f"Mean distance to origin: {mean_norm:.1e} ± {std_norm:.1e}")


def orthonormalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Copy to not modify the original points.
    pts_orth = pts.copy()
    # Center.
    pts_orth -= np.mean(pts_orth, 0)
    # Homogenize.
    cov = get_covariance_matrix(pts_orth, center=True, normalize=False)
    cov = scipy.linalg.sqrtm(np.linalg.inv(cov))
    pts_orth = np.array([np.real(cov.dot(x)) for x in pts_orth])
    # Normalize.
    mean_norm = np.mean(np.linalg.norm(pts_orth, axis=1))
    pts_orth /= mean_norm
    cov /= mean_norm
    return pts_orth, cov


"""---------------------------------------------------------------------------------------------------------------------
Lie PCA
---------------------------------------------------------------------------------------------------------------------"""


def get_lie_pca_operator(
    pts: np.ndarray,
    nb_neighbors: int,
    orbit_dim: int,
    method: str = "PCA",
    correction: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """
    Computes the Lie-PCA operator of a given point cloud. For the estimation of normal spaces, two options are available:
            - 'covariance': uses local covariance matrices, correctly normalized.
            - 'PCA': uses local PCA, i.e., takes the top eigenvectors of previous estimation).
    In addition, the parameter 'correction' can be set to True to ensure that the operator is the identity on symmetric
    matrices (hence its kernel contains only skew-symmetric matrices).

    Args:
        pts (np.ndarray): Array of shape (nb_points, ambient_dim) representing the sampled points on the orbit.
        nb_neighbors (int): Number of neighbors to use for tangent space estimation.
        orbit_dim (int): Dimension of the orbit for tangent space estimation.
        method (str, optional): Method to estimate normal spaces, either 'covariance' or 'PCA'. Defaults to 'PCA'.
        correction (bool, optional): If True, applies a correction to ensure the operator is the identity on symmetric
            matrices. Defaults to True.
        verbose (bool, optional): If True, prints eigenvalues and eigengap information. Defaults to False.

    Returns:
        np.ndarray: The Lie-PCA operator as a matrix of shape (ambient_dim**2, ambient_dim**2).
    """
    nb_points, ambient_dim = np.shape(pts)
    # Compute local covariance matrices.
    kdt = sklearn.neighbors.KDTree(pts, leaf_size=nb_neighbors + 1, metric="euclidean")
    neighbors_idx = kdt.query(pts, nb_neighbors + 1, return_distance=False)[:, 1::]
    proj_tangent_spaces = [
        get_covariance_matrix(
            pts=pts[i] - pts[neighbors_idx[i, :]],
            center=False,
            normalize=True,
            orbit_dim=orbit_dim,
        )
        for i in range(nb_points)
    ]
    # Compute projection on normal spaces via local covariance (take complementary of previous estimation).
    if method == "covariance":
        proj_normal_spaces = [
            np.eye(ambient_dim) - proj for proj in proj_tangent_spaces
        ]
    # Compute projection on normal spaces via local PCA.
    elif method == "PCA":
        proj_normal_spaces = []
        for proj_tangent in proj_tangent_spaces:
            # Get the eigenvalues and eigenvectors of the covariance matrix.
            eigenvalues, mat = scipy.linalg.eigh(proj_tangent)
            # Get top "orbit_dim" indices (largest eigenvalues). They represent the tangent space.
            idx = np.argsort(eigenvalues)[-orbit_dim:]
            # Create canonical projection matrix (zero out the tangent space).
            proj_normal = np.eye(ambient_dim)
            proj_normal[idx, idx] = 0
            # Conjugate.
            proj_normal_spaces.append(mat @ proj_normal @ mat.T)
    else:
        raise ValueError(f"Method {method} not recognized.")
    # Compute projections on lines.
    proj_lines = [
        np.outer(pts[i, :], pts[i, :]) / np.dot(pts[i, :], pts[i, :])
        for i in range(nb_points)
    ]
    # Create basis of space of matrices.
    basis_matrices = []
    for i in range(ambient_dim):
        for j in range(ambient_dim):
            mat = np.zeros((ambient_dim, ambient_dim))
            mat[i, j] = 1
            basis_matrices.append(mat)
    # Compute Lie-PCA operator.
    lie_pca = np.zeros((ambient_dim**2, ambient_dim**2))
    for i in range(len(basis_matrices)):
        lie_pca[:, i] = np.sum(
            [
                proj_normal_spaces[j] @ basis_matrices[i] @ proj_lines[j]
                for j in range(nb_points)
            ],
            axis=0,
        ).flatten()
    lie_pca /= len(pts)
    # Correction: set values of non-skew-symmetric matrices to zero. To do so, we skew-symmetrize the basis.
    if correction:
        lie_pca_corrected = np.zeros((ambient_dim**2, ambient_dim**2))
        for k in range(len(basis_matrices)):
            # Take basis element and decompose it into symmetric and skew-symmetric parts.
            mat = basis_matrices[k]
            mat_sym = (mat + mat.T) / 2
            mat_skew_sym = (mat - mat.T) / 2
            # Compute the image via Lie PCA. The image of the symmetric part is itself, so that its eigenvalue is 1.
            mat_sym_image = mat_sym
            mat_skew_sym_image = (lie_pca @ (mat_skew_sym.reshape(-1))).reshape(
                ambient_dim, ambient_dim
            )
            mat_image = mat_sym_image + mat_skew_sym_image
            # Store the image in the Lie PCA operator.
            lie_pca_corrected[:, k] = mat_image.flatten()
        # Symmetrize the operator.
        lie_pca_corrected = (lie_pca_corrected + lie_pca_corrected.T) / 2
        lie_pca = lie_pca_corrected
    # Print eigenvalues and eigengap.
    if verbose:
        vals = np.sort(np.linalg.eigvals(lie_pca).real)
        print("Lie PCA first eigenvalues:", *[f"{v:.1e} " for v in vals[:4]], end=" ")
        print(
            f"\x1b[34mEigengap #{orbit_dim}: {(vals[orbit_dim] / vals[orbit_dim - 1]):.1e}\x1b[0m."
        )
    return lie_pca
