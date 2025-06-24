# Third-party imports.
import sklearn
import matplotlib.pyplot as plt
import matplotlib

# Local imports.
from algebra import are_representations_equivalent
from liepca import get_lie_pca_operator
from orbits import (
    sample_orbit_from_group,
    sample_orbit_from_rep,
    sample_orbit_from_algebra,
    print_hausdorff_distance,
)
from optimization import find_closest_algebra


# Set matplotlib.
matplotlib.use("TkAgg")
plot_figures = False


" Optimization - SO(2) in R^4 "
print("\nOptimization - SO(2) in R^4")


# Generate orbit.
group = "torus"
nb_points = 500
group_dim = 1
method = "random"
lattice = ((1, 4),)
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method="random",
    conjugate_algebra=True,
    translate_orbit=True,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 20
orbit_dim = 1
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Find closest pushforward algebra.
frequency_max = 4
for method in ["full_lie_pca", "bottom_lie_pca", "abelian"]:
    # Optimization.
    optimal_rep, optimal_algebra = find_closest_algebra(
        group=group,
        lie_pca=lie_pca,
        group_dim=group_dim,
        frequency_max=frequency_max,
        span_ambient_space=True,
        method=method,
        verbose=True,
        verbose_top_scores=False,
    )

    # Sanity check: orbit-equivalence of reps.
    are_representations_equivalent(
        group=group, rep0=lattice, rep1=optimal_rep, verbose=True
    )

    # Sanity check: Hausdorff distance.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=optimal_rep,
        algebra=optimal_algebra,
        x=pts[0],
        nb_points=1000,
        method="uniform",
        verbose=False,
    )
    print_hausdorff_distance(pts, orbit)

    # Plot orbit.
    if plot_figures:
        pca = sklearn.decomposition.PCA(n_components=3).fit(pts)
        pts_pca = pca.transform(pts)
        orbit_pca = pca.transform(orbit)
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.scatter(pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2], c="black", s=10)
        ax.plot(
            orbit_pca[:, 0],
            orbit_pca[:, 1],
            orbit_pca[:, 2],
            c="magenta",
            lw=10,
            alpha=0.5,
        )
        plt.show()


" Optimization - SO(2) in R^8 "
print("\nOptimization - SO(2) in R^8")


# Generate orbit.
group = "torus"
nb_points = 1000
group_dim = 1
method = "random"
lattice = ((1, 5, 3, -2),)
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method="random",
    conjugate_algebra=True,
    translate_orbit=True,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 20
orbit_dim = 1
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Find closest pushforward algebra.
frequency_max = 6
for method in ["abelian"]:
    # Optimization.
    optimal_rep, optimal_algebra = find_closest_algebra(
        group=group,
        lie_pca=lie_pca,
        group_dim=group_dim,
        frequency_max=frequency_max,
        span_ambient_space=True,
        method=method,
        verbose=True,
        verbose_top_scores=False,
    )

    # Sanity check: orbit-equivalence of reps.
    are_representations_equivalent(
        group=group, rep0=lattice, rep1=optimal_rep, verbose=True
    )

    # Sanity check: Hausdorff distance.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=optimal_rep,
        algebra=optimal_algebra,
        x=pts[0],
        nb_points=1000,
        method="uniform",
        verbose=False,
    )
    print_hausdorff_distance(pts, orbit)

    # Plot orbit.
    if plot_figures:
        pca = sklearn.decomposition.PCA(n_components=3).fit(pts)
        pts_pca = pca.transform(pts)
        orbit_pca = pca.transform(orbit)
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.scatter(pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2], c="black", s=10)
        ax.plot(
            orbit_pca[:, 0],
            orbit_pca[:, 1],
            orbit_pca[:, 2],
            c="magenta",
            lw=10,
            alpha=0.5,
        )
        plt.show()


" Optimization - T^2 "
print("\nOptimization - T^2")


# Generate orbit.
group = "torus"
nb_points = 2000
group_dim = 2
method = "random"
lattice = ((2, -2, 1), (2, 0, 1))
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method=method,
    conjugate_algebra=True,
    translate_orbit=True,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 40
orbit_dim = 2
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Find closest pushforward algebra.
frequency_max = 2
for method in ["bottom_lie_pca", "abelian"]:
    # Optimization.
    optimal_rep, optimal_algebra = find_closest_algebra(
        group=group,
        lie_pca=lie_pca,
        group_dim=group_dim,
        frequency_max=frequency_max,
        span_ambient_space=True,
        method=method,
        verbose=True,
        verbose_top_scores=False,
    )

    # Sanity check: orbit-equivalence of reps.
    are_representations_equivalent(
        group=group, rep0=lattice, rep1=optimal_rep, verbose=True
    )

    # Sanity check: Hausdorff distance.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=optimal_rep,
        algebra=optimal_algebra,
        x=pts[0],
        nb_points=150**2,
        method="uniform",
        verbose=False,
    )
    print_hausdorff_distance(pts, orbit)

    # Plot orbit.
    if plot_figures:
        pca = sklearn.decomposition.PCA(n_components=3).fit(pts)
        pts_pca = pca.transform(pts)
        orbit_pca = pca.transform(orbit)
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.scatter(pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2], c="black", s=10)
        ax.scatter(
            orbit_pca[:, 0],
            orbit_pca[:, 1],
            orbit_pca[:, 2],
            c="magenta",
            lw=0.5,
            alpha=0.25,
            marker="3",
        )
        plt.show()


" Optimization - T^3 "
print("\nOptimization - T^3")


# Generate orbit.
group = "torus"
nb_points = 5000
group_dim = 3
method = "random"
lattice = ((-1, 1, 0, 1), (-2, 1, -1, 2), (-2, 1, -2, 0))
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method="random",
    conjugate_algebra=True,
    translate_orbit=True,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 50
orbit_dim = 3
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Find closest pushforward algebra.
frequency_max = 2
for method in ["abelian"]:
    # Optimization.
    optimal_rep, optimal_algebra = find_closest_algebra(
        group=group,
        lie_pca=lie_pca,
        group_dim=group_dim,
        frequency_max=frequency_max,
        span_ambient_space=True,
        method=method,
        verbose=True,
        verbose_top_scores=False,
    )

    # Sanity check: orbit-equivalence of reps.
    are_representations_equivalent(
        group=group, rep0=lattice, rep1=optimal_rep, verbose=True
    )

    # Sanity check: Hausdorff distance.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=optimal_rep,
        algebra=optimal_algebra,
        x=pts[0],
        nb_points=30**3,
        method="uniform",
        verbose=False,
    )
    print_hausdorff_distance(pts, orbit)

    # Plot orbit.
    if plot_figures:
        pca = sklearn.decomposition.PCA(n_components=3).fit(pts)
        pts_pca = pca.transform(pts)
        orbit_pca = pca.transform(orbit)
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.scatter(pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2], c="black", s=10)
        ax.scatter(
            orbit_pca[:, 0],
            orbit_pca[:, 1],
            orbit_pca[:, 2],
            c="magenta",
            lw=0.5,
            alpha=0.25,
            marker="3",
        )
        plt.show()


" Optimization - SU(2) "
print("\nOptimization - SU(2)")


# Generate orbit.
group = "SU(2)"
ambient_dim = 7
nb_points = 20**3
method = "random"
pts, rep_type = sample_orbit_from_group(
    group=group,
    ambient_dim=ambient_dim,
    nb_points=nb_points,
    frequency_max=None,
    group_dim=None,
    conjugate_algebra=True,
    right_multiply_algebra=True,
    translate_orbit=True,
    method=method,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 30
orbit_dim = 3
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Find closest pushforward algebra.
for method in ["full_lie_pca", "bottom_lie_pca"]:
    # Optimization.
    optimal_rep, optimal_algebra = find_closest_algebra(
        group=group,
        lie_pca=lie_pca,
        group_dim=None,
        frequency_max=None,
        span_ambient_space=False,
        method=method,
        verbose=True,
        verbose_top_scores=True,
    )

    # Sanity check: orbit-equivalence of reps.
    are_representations_equivalent(
        group=group, rep0=rep_type, rep1=optimal_rep, verbose=True
    )

    # Sanity check: Hausdorff distance.
    orbit = sample_orbit_from_algebra(
        group=group,
        rep_type=optimal_rep,
        algebra=optimal_algebra,
        x=pts[0],
        nb_points=30**3,
        method="uniform",
        verbose=False,
    )
    print_hausdorff_distance(pts, orbit)

    # Plot orbit.
    if plot_figures:
        pca = sklearn.decomposition.PCA(n_components=3).fit(pts)
        pts_pca = pca.transform(pts)
        orbit_pca = pca.transform(orbit)
        _, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        ax.scatter(pts_pca[:, 0], pts_pca[:, 1], pts_pca[:, 2], c="black", s=10)
        ax.scatter(
            orbit_pca[:, 0],
            orbit_pca[:, 1],
            orbit_pca[:, 2],
            c="magenta",
            lw=0.5,
            alpha=0.25,
            marker="3",
        )
        plt.show()
