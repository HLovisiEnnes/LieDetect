"""---------------------------------------------------------------------------------------------------------------------

LieDetect: Detection of representation orbits of compact Lie groups from point clouds
Henrique Ennes & Raphaël Tinarrage
See the repo at https://github.com/HLovisiEnnes/LieDetect and the article at https://arxiv.org/abs/2309.03086

------------------------------------------------------------------------------------------------------------------------

This module provides functions to sample points on orbits of representations, from their Lie algebra generators. Namely,
the Lie algebras are stored through bases (tuples of skew-symmetric matrices), that we suppose to be isomorphic to
the canonical pushforward algebras of the representations implemented in the module "algebra", through conjugation by an
orthogonal matrix. This assumption it crucial since, combined with the representation type of the algebra, we are able
to compute the periods of the basis elements (minimal t>0 such that exp(tA)=I), and hence be able to reconstruct the
orbit entirely (and without repetition for the torus). Note that, given the algebra alone, we cannot compute the periods
in a robust way.

------------------------------------------------------------------------------------------------------------------------

Sample on orbits:
    print_hausdorff_distance
    get_periods
    sample_orbit_from_algebra
    sample_orbit_from_group

---------------------------------------------------------------------------------------------------------------------"""

# Standard imports.
import math
from typing import Optional, List

# Third-party imports.
import numpy as np
import scipy

# Local imports.
from algebra import (
    get_random_lattice,
    get_random_constrained_partition,
    get_canonical_pushforward_algebra,
)


"""---------------------------------------------------------------------------------------------------------------------
Sample on orbits
---------------------------------------------------------------------------------------------------------------------"""


def print_hausdorff_distance(pts1: np.ndarray, pts2: np.ndarray) -> None:
    """Prints the Hausdorff distance from first point cloud to second."""
    hausdorff_dist = scipy.spatial.distance.directed_hausdorff(pts1, pts2)[0]
    print(f"Non-symmetric \x1b[34mHausdorff distance: {hausdorff_dist:.3e}\x1b[0m.")
    return hausdorff_dist


def get_periods(
    group: str,
    rep_type: tuple,
    algebra: List[np.ndarray],
) -> List[float]:
    """
    Returns a list of minimal periods t>0 such that exp(tA)=I for each A in the pushforward algebra. We suppose that the
    elements in "algebra" are such that they generate a periodic 1-parameter subgroup. More precisely, we suppose that
    the bijection between "algebra" and the canonical algebra (implemented in get_canonical_pushforward_algebra) is a
    Lie algebra isomorphism, induced by a conjugation. In particular, up to normalization by the norm of the elements,
    the periods are identical.

    Case of SO(2):
        The period of the integer matrix A* representing the pushforward algebra of SO(2) via the rep (a1, ..., am) is
                2 * pi / gcd(a1, ..., am).
        Its norm is
                sqrt(2) * ||(a1, ..., am)||.
        In particular, if A is a skew-symmetric matrix that is, up to normalization, conjugate to A*, then its period is
                2 * pi / gcd(a1, ..., am) * 2 * ||(a1, ..., am)|| / ||A||.

    Case of T^d:
        Similar to the case of SO(2), but reasoning coordinate by coordinate.

    Case of SU(2):
        The period of the canonical matrices representing the algebra of irrep 2j+1 (integer) or 4j+2 (half-integer) is
                2 * pi   or   4 * pi, respectively.
        Their norm is
                n(j) = sqrt(j * (j + 1) * (2 * j + 1) / 3)   or   sqrt(j * (j + 1) * (4 * j + 2) / 3), respectively.
        If A* is a canonical matrix coming from the sum of irreps (d1, ..., dm), then its period is
                4 * pi if there exists a half integer, 2 * pi otherwise,
        and its norm is the combination of the norms above.
        In particular, if A is a skew-symmetric matrix that is, up to normalization, conjugate to A^*, then its period is
                (2 or 4) * pi * ||(n(j1), ..., n(jm))|| / ||A||.
    """

    def period_so2(weights: tuple[int, ...]) -> float:
        return 2 * np.pi / math.gcd(*weights)

    def norm_so2(weights: tuple[int, ...]) -> float:
        return np.sqrt(2) * np.linalg.norm(weights)

    def period_irrep_su2(dim: int) -> float:
        if dim % 4 == 0:
            return 4 * np.pi
        else:
            return 2 * np.pi

    def norm_irrep_su2(dim: int) -> float:
        if dim % 4 == 0:
            j = (dim - 2) / 4
            return np.sqrt(j * (j + 1) * (4 * j + 2) / 3)
        else:
            j = (dim - 1) / 2
            return np.sqrt(j * (j + 1) * (2 * j + 1) / 3)

    # Compute the periods based on the group type.
    if group == "torus":
        periods = [
            period_so2(weights) * norm_so2(weights) / np.linalg.norm(mat)
            for weights, mat in zip(rep_type, algebra)
        ]
    elif group in ["SU(2)", "SO(3)"]:
        period = max(period_irrep_su2(dim) for dim in rep_type)
        norm = np.linalg.norm([norm_irrep_su2(dim) for dim in rep_type])
        periods = [period * norm / np.linalg.norm(mat) for mat in algebra]
    else:
        raise NotImplementedError(f"Group '{group}' not recognized.")
    # Sanity check: the exponentiated matrices should be the identity.
    for period, mat in zip(periods, algebra):
        if not np.isclose(scipy.linalg.expm(period * mat), np.eye(len(mat))).all():
            print(
                "Error! Incorrect period. Distance to identity:",
                np.linalg.norm(scipy.linalg.expm(period * mat) - np.eye(len(mat))),
            )
    return periods


def sample_orbit_from_algebra(
    group: str,
    rep_type: tuple,
    algebra: List[np.ndarray],
    x: np.ndarray,
    nb_points: int,
    method: str = "uniform",
    verbose: bool = False,
) -> np.ndarray:
    """
    Samples points on the orbit of a compact Lie group representation, given its Lie algebra generators. We suppose that
    the algebra is isomorphic to the canonical algebra indicated in rep_type. This allows us to compute the periods,
    which are, otherwise, not stably computable from the algebra alone.

    Note that the output orbit, in the uniform case, may not contain exactly nb_points points (this only happens if the
    parameter nb_points is a perfect power of the group dimension). In the random case, it will contain exactly
    nb_points points.

    Args:
        group (str): The group type, e.g., 'torus' or 'SU(2)'.
        rep_type (tuple): Representation type parameters (e.g., weights or partition).
        algebra (List[np.ndarray]): List of Lie algebra generators as matrices.
        x (np.ndarray): Initial vector to act on.
        nb_points (int): Number of points to sample.
        method (str): Sampling method, 'uniform' or 'random'. Defaults to 'uniform'.
        verbose (bool): Whether to print information about the sampled orbit.

    Returns:
        np.ndarray: Array of sampled points on the orbit.
    """
    # Get periods.
    periods = get_periods(group, rep_type, algebra)
    # Generate orbit based on the method.
    if method == "uniform":
        # Get number of points.
        group_dim = len(algebra)
        nb_points_circle = int(nb_points ** (1 / group_dim))
        # Generate first circle.
        times = np.linspace(0, periods[0], nb_points_circle)
        orbit = np.array([scipy.linalg.expm(t * algebra[0]) @ x for t in times])
        # Apply next transformations
        for i in range(1, len(algebra)):
            times = np.linspace(0, periods[i], nb_points_circle)
            orbit = [
                [scipy.linalg.expm(t * algebra[i]) @ y for t in times] for y in orbit
            ]
            orbit = np.concatenate(orbit)
    elif method == "random":
        # Generate random points in hypercube [0, period[0]] x ... x [0, period[-1]].
        periods = np.asarray(periods)
        times = np.random.rand(nb_points, periods.size) * periods
        # Generate orbit (linear combinations or algebra elements wrt times).
        orbit = np.array(
            [
                scipy.linalg.expm(
                    np.sum([t * mat for t, mat in zip(times[i], algebra)], axis=0)
                )
                @ x
                for i in range(nb_points)
            ]
        )
    else:
        raise ValueError(
            f"Method '{method}' not recognized. Use 'uniform' or 'random'."
        )
    # Print comments.
    if verbose:
        print(
            f"Sampled {len(orbit)} {method} points on the orbit of \x1b[1;31m{group} with rep {rep_type}\x1b[0m."
        )
    return orbit


def sample_orbit_from_rep(
    group: str,
    rep_type: tuple,
    nb_points: int,
    conjugate_algebra: bool = False,
    right_multiply_algebra: bool = False,
    translate_orbit: bool = False,
    method: str = "random",
    verbose: bool = False,
) -> np.ndarray:
    ambient_dim = len(rep_type[0]) * 2 if group == "torus" else sum(rep_type)
    # Get canonical pushforward algebra.
    algebra = get_canonical_pushforward_algebra(group=group, rep_type=rep_type)
    # Get initial vector.
    x = np.ones(ambient_dim)
    x /= np.linalg.norm(x)
    # Conjugate algebra if needed.
    if conjugate_algebra:
        orth = scipy.stats.special_ortho_group.rvs(ambient_dim)
        algebra = [orth @ mat @ orth.T for mat in algebra]
        # Translate initial vector (to conserve a homogeneous orbit).
        x = orth @ x
    # Right-multiply algebra if needed.
    if right_multiply_algebra:
        if group == "torus":
            raise NotImplementedError(
                "Right-multiply not implemented for torus representations."
            )
        orth = scipy.stats.special_ortho_group.rvs(3)
        algebra = [
            np.sum([algebra[j] * orth[j, i] for j in range(len(algebra))], axis=0)
            for i in range(len(algebra))
        ]
    # Sample orbit.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=rep_type,
        algebra=algebra,
        x=x,
        nb_points=nb_points,
        method=method,
        verbose=verbose,
    )
    # Translate orbit if needed.
    if translate_orbit:
        orth = scipy.stats.special_ortho_group.rvs(ambient_dim)
        orbit = np.array([orth @ point for point in orbit])
    return orbit


def sample_orbit_from_group(
    group: str,
    ambient_dim: int,
    nb_points: int,
    frequency_max: Optional[int] = None,
    group_dim: Optional[int] = None,
    conjugate_algebra: bool = False,
    right_multiply_algebra: bool = False,
    translate_orbit: bool = False,
    method: str = "random",
    span_ambient_space: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, tuple]:
    """
    Samples an orbit of a random representation in a given ambient space.

    Args:
        group (str): The group type, e.g., 'torus', 'SU(2)', or 'SO(3)'.
        ambient_dim (int): Dimension of the ambient space.
        nb_points (int): Number of points to sample on the orbit.
        frequency_max (Optional[int]): Maximal frequency for torus representations.
        group_dim (Optional[int]): Dimension of the group (for torus).
        conjugate_algebra (bool): Whether to conjugate the algebra by a random orthogonal matrix.
        right_multiply_algebra (bool): Whether to right-multiply the algebra by a random orthogonal matrix.
        translate_orbit (bool): Whether to translate the sampled orbit by a random orthogonal transformation.
        method (str): Sampling method, 'random' or 'uniform'.
        span_ambient_space (bool): Whether to only consider representations whose orbits span the ambient space. Only
            implemented for the circle or the non-Abelian groups.
        verbose (bool): Whether to print information about the sampled orbit.

    Returns:
        tuple[np.ndarray, tuple]: The sampled orbit (array of shape (nb_points, ambient_dim)) and representation type.
    """
    # Get random representation.
    if group == "torus":
        rep_type = get_random_lattice(
            lattice_rank=group_dim,
            ambient_rank=ambient_dim // 2,
            frequency_max=frequency_max,
            span_ambient_space=span_ambient_space,
        )
    elif group in ["SU(2)", "SO(3)"]:
        rep_type = get_random_constrained_partition(
            group=group, ambient_dim=ambient_dim, span_ambient_space=span_ambient_space
        )
    else:
        raise NotImplementedError(f"Group '{group}' recognized.")
    # Sample orbit from the representation type.
    orbit = sample_orbit_from_rep(
        group=group,
        rep_type=rep_type,
        nb_points=nb_points,
        conjugate_algebra=conjugate_algebra,
        right_multiply_algebra=right_multiply_algebra,
        translate_orbit=translate_orbit,
        method=method,
        verbose=verbose,
    )
    return orbit, rep_type
