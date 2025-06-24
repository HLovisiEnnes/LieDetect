"""---------------------------------------------------------------------------------------------------------------------

LieDetect: Detection of representation orbits of compact Lie groups from point clouds
Henrique Ennes & Raphaël Tinarrage
See the repo at https://github.com/HLovisiEnnes/LieDetect and the article at https://arxiv.org/abs/2309.03086

------------------------------------------------------------------------------------------------------------------------

Implementation of the optimization methods to find the closest Lie algebra to a given Lie PCA operator, according to
three methods:
    "full_lie_pca": seeks a pushforward algebra that minimizes Lie PCA operator.
    "bottom_lie_pca": seeks a pushforward algebra close to the bottom eigenvectors of the Lie PCA operator.
    "abelian": computes the normal for of the bottom eigenvectors of the Lie PCA operator and finds the closest lattice.

------------------------------------------------------------------------------------------------------------------------

Utils:
    chronometer_start
    chronometer_tick

Closest Lie algebra - general case:
    PARAMS_OPTIMIZATION
    find_closest_algebra
    optimization_bottom_lie_pca
    optimization_full_lie_pca

Closest Lie algebra - Abelian case:
    optimization_abelian
    normal_form_skew_sym_matrix
    normal_form_skew_symmetric_matrices

---------------------------------------------------------------------------------------------------------------------"""

# Standard imports.
from typing import Optional, List
import itertools
import time, datetime, sys

# Third-party imports.
import autograd.numpy as np
import pymanopt, scipy

# Local imports.
from algebra import (
    skew_sym_frame_to_projection,
    get_lattices,
    invariant_of_lattices,
    get_constrained_partitions,
    get_canonical_pushforward_algebra,
)


"""---------------------------------------------------------------------------------------------------------------------
Utils
---------------------------------------------------------------------------------------------------------------------"""


def chronometer_start(msg: str = "Start... ", verbose: bool = True) -> float:
    start_time = time.time()
    if verbose:
        sys.stdout.write(msg)
        sys.stdout.flush()
    return start_time


def chronometer_tick(start_time: float, i: int, i_total: int, msg: str) -> None:
    elapsed_time_secs = time.time() - start_time
    expected_time_secs = (i_total - i - 1) / (i + 1) * elapsed_time_secs
    msg1 = "It " + repr(i + 1) + "/" + repr(i_total) + ". "
    msg2 = "Duration %s. " % datetime.timedelta(seconds=round(elapsed_time_secs))
    msg3 = "Remaining %s." % datetime.timedelta(seconds=round(expected_time_secs))
    sys.stdout.write("\r" + msg + msg1 + msg2 + msg3)
    if i >= i_total - 1:
        sys.stdout.write("\n")


# def ChronometerStop(start_time, method="ms", linebreak="\n"):
#     elapsed_time_secs = time.time() - start_time
#     if method == "ms":
#         msg = (
#             "Execution time: "
#             + repr(round(elapsed_time_secs * 1000))
#             + " ms."
#             + linebreak
#         )
#     if method == "s":
#         msg = "Execution time: " + repr(round(elapsed_time_secs)) + " s." + linebreak
#     sys.stdout.write(msg)
#     sys.stdout.flush()


"""---------------------------------------------------------------------------------------------------------------------
Closest Lie algebra - general case
---------------------------------------------------------------------------------------------------------------------"""


PARAMS_PYMANOPT = {
    "max_iterations": 100,
    "min_gradient_norm": 1e-6,
    "max_time": 300,  # in seconds
    "verbosity": 0,
}


def find_closest_algebra(
    group: str,
    lie_pca: np.ndarray,
    group_dim: Optional[int] = None,
    frequency_max: Optional[int] = None,
    reps_to_test=None,
    span_ambient_space: bool = True,
    method="bottom_lie_pca",
    verbose: bool = True,
    verbose_top_scores: bool = False,
) -> (tuple, List[np.ndarray]):
    """
    Finds the closest pushforward Lie algebra of a group (subspace of skew-symmetric matrices) to the given Lie PCA
    operator.

    Args:
        group (str): The group ('torus', 'SU(2)', or 'SO(3)').
        lie_pca (np.ndarray): The Lie PCA operator (matrix).
        group_dim (Optional[int]): Dimension of the torus (if group='torus'), otherwise ignored.
        frequency_max (int): Maximum frequency for lattice search (torus case).
        reps_to_test (list, optional): List of representations to test. If None, computed automatically.
        span_ambient_space (bool): Whether to restrict to representations with orbits spanning the ambient space.
        method (str): Optimization method: 'bottom_lie_pca', 'full_lie_pca', or 'abelian' (torus only).
        verbose (bool): If True, print progress and results.
        verbose_top_scores (bool): If True, print top scoring representations.

    Returns:
        optimal_rep: The optimal representation type found.
        optimal_algebra: List of matrices forming the optimal Lie algebra.
    """
    group_dim = group_dim if group == "torus" else 3
    ambient_dim = int(np.sqrt(lie_pca.shape[0]))
    # Print comments.
    if verbose:
        sys.stdout.write(
            f"----> Optimization problem via \x1b[34m{method}\x1b[0m method for {group}  <----\n"
        )
    # Get representations to test.
    if reps_to_test is None:
        if group == "torus":
            # For the abelian method, we only take lattices up to span-equivalence, not orbit-equivalence,
            # for they are faster to compute.
            reps_to_test = get_lattices(
                lattice_rank=group_dim,
                ambient_rank=int(ambient_dim / 2),
                frequency_max=frequency_max,
                # method=(
                #     "orbit-equivalence" if method == "abelian" else "orbit-equivalence"
                # ),
                method="orbit-equivalence",
                span_ambient_space=span_ambient_space,
                verbose=False,
            )
        elif group in ["SU(2)", "SO(3)"]:
            reps_to_test = get_constrained_partitions(
                group=group,
                ambient_dim=ambient_dim,
                span_ambient_space=span_ambient_space,
            )
        else:
            raise ValueError(f"Group {group} not recognized.")
    # Get bottom eigenvectors of Lie PCA if needed.
    if method in ["bottom_lie_pca", "abelian"]:
        vals, vecs = np.linalg.eigh(lie_pca)
        vecs = [
            vecs[:, i].reshape((ambient_dim, ambient_dim)) for i in range(group_dim)
        ]
        lie_pca_algebra = [(A - A.T) / 2 for A in vecs]
        lie_pca_proj = skew_sym_frame_to_projection(lie_pca_algebra)
    # Run optimization in the general case.
    if method in ["bottom_lie_pca", "full_lie_pca"]:
        if verbose:
            start_time = chronometer_start("Solve minimization problem... ")
        costs = dict()
        minimizers = dict()
        # Run over all rep types.
        for i in range(len(reps_to_test)):
            rep_type = reps_to_test[i]
            # Run over the two connected components of the orthogonal group.
            for determinant in ["+1", "-1"]:
                if method == "bottom_lie_pca":
                    result = optimization_bottom_lie_pca(
                        group=group,
                        lie_pca_proj=lie_pca_proj,
                        rep_type=rep_type,
                        determinant=determinant,
                    )
                elif method == "full_lie_pca":
                    result = optimization_full_lie_pca(
                        group=group,
                        lie_pca=lie_pca,
                        rep_type=rep_type,
                        determinant=determinant,
                    )
                costs[(rep_type, determinant)] = result.cost
                minimizers[(rep_type, determinant)] = result.point
            if verbose:
                chronometer_tick(
                    start_time,
                    i,
                    len(reps_to_test),
                    "Solve minimization problem... ",
                )
        # Define optimal algebra.
        optimal_rep, optimal_det = min(costs.keys(), key=(lambda k: costs[k]))
        optimal_change_of_basis = minimizers[(optimal_rep, optimal_det)]
        canonical_algebra = get_canonical_pushforward_algebra(
            group=group, rep_type=optimal_rep
        )
        optimal_algebra = [
            optimal_change_of_basis @ mat @ optimal_change_of_basis.T
            for mat in canonical_algebra
        ]
    # Run optimization in the Abelian case.
    elif method == "abelian":
        optimal_rep, optimal_algebra, costs = optimization_abelian(
            lie_pca_algebra=lie_pca_algebra, reps_to_test=reps_to_test
        )
    # Print comments.
    if verbose:
        print(
            f"The optimal rep found is \x1b[1;31m{optimal_rep}\x1b[0m with cost {min(costs.values()):.3e}."
        )
    if verbose_top_scores:
        nb_scores_to_print = 10
        top_scores = sorted(costs.items(), key=lambda x: x[1])[:nb_scores_to_print]
        for i, (rep, score) in enumerate(top_scores):
            print(f"    {rep} - cost {score:.3e} (best cost #{i + 1}/{len(costs)})")
    return optimal_rep, optimal_algebra


def optimization_bottom_lie_pca(
    group: str,
    lie_pca_proj: np.ndarray,
    rep_type: tuple,
    determinant: str = "+1",
):
    """
    Given a representation type and an initial guess of pushforward algebra, this function optimizes over the special
    orthogonal matrices to find a conjugation of the canonical matrices such that is the closest to the initial guess.
    We encode algebras (subspaces of the skew-symmetric matrices) as projection matrices.

    Args:
        group (str): The group type ('torus', 'SU(2)', or 'SO(3)').
        lie_pca_proj (np array): The target projection matrix on the space of skew-symmetric matrices.
        rep_type (tuple): The representation type (e.g., frequencies or partition).
        determinant (int): '+1' for SO(n), '-1' for the other component of O(n).

    Returns:
        result: A pymanopt optimization result with the optimal orthogonal matrix and cost.
    """
    # Define basis.
    canonical_algebra = get_canonical_pushforward_algebra(
        group=group, rep_type=rep_type
    )
    # Define cost function.
    ambient_dim = len(canonical_algebra[0])
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def _cost_function(orth: np.ndarray) -> float:
        # Transform to provided connected component of orthogonal group.
        if determinant == "-1":
            orth = orth @ np.diag([-1] + [1] * (ambient_dim - 1))
        # Conjugate canonical matrices.
        algebra = [orth @ mat @ orth.T for mat in canonical_algebra]
        # Compute projection matrix.
        proj = skew_sym_frame_to_projection(algebra, method="differentiable")
        # Compute distance to objective-
        difference = proj - lie_pca_proj
        dist = np.trace(difference @ difference.T)
        return dist

    # Run optimization.
    problem = pymanopt.Problem(manifold, _cost_function)
    optimizer = pymanopt.optimizers.SteepestDescent(**PARAMS_PYMANOPT)
    result = optimizer.run(problem)
    # Transform to provided connected component of orthogonal group.
    if determinant == "-1":
        result.point = result.point @ np.diag([-1] + [1] * (ambient_dim - 1))
    return result


def optimization_full_lie_pca(
    group: str,
    lie_pca: np.ndarray,
    rep_type: tuple,
    determinant: str = "+1",
):
    """
    Given a representation type and the Lie PCA operator, this function optimizes over the special orthogonal matrices
    to find a conjugation of the canonical matrices that is the closest to the kernel of Lie PCA.

    Args:
        group (str): The group type ('torus', 'SU(2)', or 'SO(3)').
        lie_pca (np array): The target projection matrix on the space of skew-symmetric matrices.
        rep_type (tuple): The representation type (e.g., frequencies or partition).
        determinant (int): '+1' for SO(n), '-1' for the other component of O(n).

    Returns:
        result: A pymanopt optimization result with the optimal orthogonal matrix and cost.
    """
    # Define basis.
    canonical_algebra = get_canonical_pushforward_algebra(
        group=group, rep_type=rep_type
    )
    # canonical_algebra = gram_schmidt_orthonormalization(canonical_algebra)
    # Define cost function.
    ambient_dim = len(canonical_algebra[0])
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def _cost_function(orth):
        # Transform to provided connected component of orthogonal group.
        if determinant == "-1":
            orth = orth @ np.diag([-1] + [1] * (ambient_dim - 1))
        # Conjugate canonical matrices
        algebra = [orth @ mat @ orth.T for mat in canonical_algebra]
        # Compute cost.
        differences = [
            lie_pca.dot(algebra[i].flatten()) for i in range(len(canonical_algebra))
        ]
        differences = [np.sum(difference @ difference.T) for difference in differences]
        return np.sum(differences)

    # Run optimization
    problem = pymanopt.Problem(manifold, _cost_function)
    optimizer = pymanopt.optimizers.SteepestDescent(**PARAMS_PYMANOPT)
    result = optimizer.run(problem)
    # Transform to provided connected component of orthogonal group.
    if determinant == "-1":
        result.point = result.point @ np.diag([-1] + [1] * (ambient_dim - 1))
    return result


"""---------------------------------------------------------------------------------------------------------------------
Closest Lie algebra - Abelian case
---------------------------------------------------------------------------------------------------------------------"""


def optimization_abelian(
    lie_pca_algebra: list, reps_to_test: list
) -> (tuple, list, dict):
    """
    Finds the closest Abelian (torus) Lie algebra to the given Lie PCA algebra.

    Args:
        lie_pca_algebra (list): List of skew-symmetric matrices representing the Lie PCA algebra.
        reps_to_test (list): List of candidate lattice representations (as arrays) to compare.

    Returns:
        optimal_rep: The optimal lattice representation found (as a tuple of tuples).
        optimal_algebra: List of matrices forming the optimal Lie algebra in the original basis.
        costs: Dictionary mapping each tested lattice (as tuple of tuples) to its cost.
    """
    group_dim = len(lie_pca_algebra)
    # # Orthonormalize the given lattice.
    # lie_pca_algebra = gram_schmidt_orthonormalization(np.array(lie_pca_algebra))
    # Find optimal weights (as real numbers) via an optimization.
    if group_dim == 1:
        optimal_weights, optimal_change_of_basis = normal_form_skew_sym_matrix(
            lie_pca_algebra[0]
        )
        optimal_weights = (optimal_weights,)
    else:
        optimal_weights, optimal_change_of_basis = normal_form_skew_symmetric_matrices(
            lie_pca_algebra
        )
    # Compute invariant for the given lattice.
    optimal_invariant = invariant_of_lattices(
        optimal_weights,
        method="span-equivalence",
        decimals_accuracy=20,
    )
    # Compare to invariant of all the lattices to test. To allow the comparison, we take each lattice (it represents an
    # orbit-equivalence class) and compute its invariant under all permutations of the columns and signs.
    m = len(optimal_weights[0])
    costs = dict()
    for weights in reps_to_test:
        if group_dim == 1:
            invariant = invariant_of_lattices(
                weights,
                method="span-equivalence",
                decimals_accuracy=20,
            )
            invariant /= np.linalg.norm(invariant)
            costs[weights] = np.linalg.norm(
                np.array(optimal_invariant) - np.array(invariant)
            )
        else:
            lattice = np.asarray(weights)
            for perm in itertools.permutations(range(m)):
                for sign in itertools.product([-1, 1], repeat=m):
                    permuted = lattice[:, perm] * sign
                    invariant = invariant_of_lattices(
                        permuted,
                        method="span-equivalence",
                        decimals_accuracy=20,
                    )
                    permuted = tuple(tuple(row) for row in permuted.tolist())
                    costs[permuted] = np.linalg.norm(
                        np.array(optimal_invariant) - np.array(invariant)
                    )
    # Deduce optimal representation.
    optimal_rep = min(costs, key=costs.get)
    # Define optimal_algebra
    optimal_algebra = get_canonical_pushforward_algebra(
        group="torus", rep_type=optimal_rep
    )
    optimal_algebra = [
        optimal_change_of_basis @ mat @ optimal_change_of_basis.T
        for mat in optimal_algebra
    ]
    return optimal_rep, optimal_algebra, costs


def normal_form_skew_sym_matrix(mat: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Compute the frequencies of the invariant planes of a skew-symmetric matrix. We do so by computing the Schur
    decomposition of the matrix, which gives us approximately a 2x2 block-diagonal matrix. These blocks are skew-sym,
    hence associated to a value, which is the frequency of the invariant plane. We then permute the entries of the
    matrix to make the frequencies increasing, and to have the positive frequencies at the top right of each block.

    Args:
        mat (np.ndarray): A real skew-symmetric matrix of shape (n, n).

    Returns:
        weights (np.ndarray): The sorted, normalized frequencies (length n/2).
        change_of_basis (np.ndarray): The orthogonal matrix that block-diagonalizes the input matrix.
    """
    # Project on the skew-symmetric matrices.
    mat = (mat - mat.T) / 2
    # Compute the Schur decomposition:  change_of_basis @ block_diag @ change_of_basis.block_diag = mat.
    block_diag, change_of_basis = scipy.linalg.schur(mat)
    # Extract the weights of the invariant planes (top right entry of each block).
    ambient_dim = np.shape(mat)[0]
    weights = [block_diag[i, i + 1] for i in range(0, ambient_dim, 2)]
    # Normalize the weights.
    weights /= np.linalg.norm(weights)
    # Find location of negative weights, and permute with the positive ones.
    index = np.where(weights < 0)[0]
    for i in index:
        weights[i] = -weights[i]
        change_of_basis[:, 2 * i + 1] *= -1
    # Sort frequencies (permute the matrix).
    index = np.argsort(weights)
    weights = weights[index]
    change_of_basis = change_of_basis[
        :, np.ravel(np.column_stack((2 * index, 2 * index + 1)))
    ]
    # index = np.argsort(weights)
    # weights = weights[index]
    # index_zip = [item for sublist in zip(2 * index, 2 * index + 1) for item in sublist]
    # change_of_basis = change_of_basis[:, index_zip]
    return weights, change_of_basis


def normal_form_skew_symmetric_matrices(
    matrices: List[np.ndarray],
) -> (np.ndarray, np.ndarray):
    """
    Computes the weights of a joint normal form of a list of skew-symmetric matrices. This is done by optimizing over
    the changes of basis (matrices in SO(n)) to cancel the entries that are not the antidiagonal terms of the 2x2
    diagonal blocks of the matrices. We assume that the matrices commute, otherwise such a joint normal form does not
    exist.

    Args:
        matrices (List[np.ndarray]): List of real skew-symmetric matrices of shape (n, n).

    Returns:
        weights (np.ndarray): Tuple of tuples containing the sorted frequencies for each matrix.
        change_of_basis (np.ndarray): The orthogonal matrix that block-diagonalizes the input matrices.
    """
    ambient_dim = np.shape(matrices[0])[0]
    m = int(np.shape(matrices[0])[0] / 2)
    # Define cost function.
    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(ambient_dim, k=1)

    @pymanopt.function.autograd(manifold)
    def _cost_function(orth):
        # Conjugate canonical matrices
        matrices_conjugate = [orth @ A @ orth.T for A in matrices]
        # Get diagonal-block terms.
        entries = [
            matrices_conjugate[i][2 * j + k, 2 * j + 1 - k]
            for i in range(len(matrices))
            for j in range(m)
            for k in (0, 1)
        ]
        entries = np.array(entries)
        # Compute norm of not block-diagonal terms.
        differences = np.array([np.trace(A @ A.T) for A in matrices_conjugate])
        norm = np.sum(differences @ differences.T) - np.sum(entries @ entries.T)
        return norm

    # Run optimization.
    problem = pymanopt.Problem(manifold, _cost_function)
    optimizer = pymanopt.optimizers.SteepestDescent(**PARAMS_PYMANOPT)
    result = optimizer.run(problem)
    change_of_basis = result.point
    # Transform in normal form.
    matrices_normal_form = [change_of_basis @ A @ change_of_basis.T for A in matrices]
    weights = tuple(
        [
            tuple([matrices_normal_form[i][2 * j + 0, 2 * j + 1] for j in range(m)])
            for i in range(len(matrices))
        ]
    )
    return weights, change_of_basis.T
