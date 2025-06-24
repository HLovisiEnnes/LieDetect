# Third-party imports.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Local imports.
from liepca import (
    get_lie_pca_operator,
    print_covariance_eigenvalues,
    project_on_minimal_subspace,
    print_norms,
    orthonormalize,
)
from orbits import sample_orbit_from_group, sample_orbit_from_rep

matplotlib.use("TkAgg")


" Lie PCA - SO(2) "
print("\nLie PCA - SO(2)")


# Generate orbit.
group = "torus"
ambient_dim = 6
nb_points = 300
frequency_max = 6
group_dim = 1
method = "random"
pts, rep_type = sample_orbit_from_group(
    group=group,
    ambient_dim=ambient_dim,
    nb_points=nb_points,
    frequency_max=frequency_max,
    group_dim=group_dim,
    conjugate_algebra=True,
    right_multiply_algebra=False,
    translate_orbit=True,
    method=method,
    span_ambient_space=True,
    verbose=True,
)

# Compute Lie-PCA operator - method "PCA".
nb_neighbors = 10
orbit_dim = 1
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=False,
    verbose=True,
)

# Plot eigenvalues Lie-PCA operator  - method "PCA".
vals = np.sort(np.linalg.eigvals(lie_pca).real)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(vals)), vals, c="black")
ax.scatter(range(len(vals)), vals, c="black")
ax.axhline(0, color="grey", linestyle="dotted")
ax.set_ylim((0 - 0.05, max(vals) + 0.05))
ax.set_title("Eigenvalues of LiePCA operator - method local PCA")
ax.set_xlabel("Index of eigenvalue")
ax.set_ylabel("Value")
plt.show()

# Compute Lie-PCA operator - method "covariance".
nb_neighbors = 10
orbit_dim = 1
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="covariance",
    correction=False,
    verbose=True,
)

# Plot eigenvalues Lie-PCA operator  - method "covariance".
vals = np.sort(np.linalg.eigvals(lie_pca).real)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(vals)), vals, c="black")
ax.scatter(range(len(vals)), vals, c="black")
ax.axhline(0, color="grey", linestyle="dotted")
ax.set_ylim((0 - 0.05, max(vals) + 0.05))
ax.set_title("Eigenvalues of LiePCA operator - method local covariance")
ax.set_xlabel("Index of eigenvalue")
ax.set_ylabel("Value")
plt.show()


" Lie PCA - T^2 "
print("\nLie PCA - T^2")


# Generate orbit.
group = "torus"
ambient_dim = 6
nb_points = 20000
frequency_max = 3
group_dim = 2
method = "random"
pts, _ = sample_orbit_from_group(
    group=group,
    ambient_dim=ambient_dim,
    nb_points=nb_points,
    frequency_max=frequency_max,
    group_dim=group_dim,
    conjugate_algebra=True,
    right_multiply_algebra=False,
    translate_orbit=True,
    method=method,
    verbose=True,
)

# Compute Lie-PCA operator.
nb_neighbors = 10
orbit_dim = 2
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=False,
    verbose=True,
)

# Plot eigenvalues Lie-PCA operator .
vals = np.sort(np.linalg.eigvals(lie_pca).real)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(vals)), vals, c="black")
ax.scatter(range(len(vals)), vals, c="black")
ax.axhline(0, color="grey", linestyle="dotted")
ax.set_ylim((0 - 0.05, max(vals) + 0.05))
ax.set_title("Eigenvalues of LiePCA operator")
ax.set_xlabel("Index of eigenvalue")
ax.set_ylabel("Value")
plt.show()


" Lie PCA - SU(2) "
print("\nLie PCA - SU(2)")

# Generate orbit.
group = "SU(2)"
ambient_dim = 5
nb_points = 50000
method = "uniform"
pts, _ = sample_orbit_from_group(
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
nb_neighbors = 100
orbit_dim = 3
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=nb_neighbors,
    orbit_dim=orbit_dim,
    method="PCA",
    correction=True,
    verbose=True,
)

# Plot eigenvalues Lie-PCA operator .
vals = np.sort(np.linalg.eigvals(lie_pca).real)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(vals)), vals, c="black")
ax.scatter(range(len(vals)), vals, c="black")
ax.axhline(0, color="grey", linestyle="dotted")
ax.set_ylim((0 - 0.05, max(vals) + 0.05))
ax.set_title("Eigenvalues of LiePCA operator")
ax.set_xlabel("Index of eigenvalue")
ax.set_ylabel("Value")
plt.show()


" Dimension reduction - SO(2) "
print("\nDimension reduction - SO(2)")


# Generate an orbit that is not ambient-dimensional.
lattice = ((2, 3, 3),)
nb_points = 200
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method="random",
    verbose=True,
)

# Print eigenvalues of covariance matrix.
print_covariance_eigenvalues(pts=pts)

# Lie PCA operator sees a large symmetry group (actually made of matrices acting trivially).
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=10,
    orbit_dim=1,
    method="PCA",
    correction=False,
    verbose=True,
)

# Project onto spanned subspace.
print("Now, project onto minimal subspace...")
pts = project_on_minimal_subspace(pts=pts, threshold_eigenvalue=1e-15)
print_covariance_eigenvalues(pts=pts)

# Now, the kernel of the Lie PCA operator is smaller.
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=10,
    orbit_dim=1,
    method="PCA",
    correction=False,
    verbose=True,
)


" Orthonormalization of non-orthogonal orbit - SO(2) "
print("\nOrthonormalization of non-orthogonal orbit - SO(2)")


# Generate an orthogonal orbit of SO(2) that spans the ambient space.
ambient_dim = 6
lattice = ((2, 3, 5),)
pts = sample_orbit_from_rep(
    group="torus",
    rep_type=lattice,
    nb_points=nb_points,
    method="random",
    verbose=True,
)
print_covariance_eigenvalues(pts)
print_norms(pts)

# Check whether orbit is orthogonal via skew-symmetricity of bottom eigenvector of Lie-PCA operator.
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=10,
    orbit_dim=1,
    method="PCA",
    correction=False,
    verbose=True,
)
eigvals, eigvecs = np.linalg.eig(lie_pca)
vec = eigvecs[:, np.argmin(eigvals)].reshape((ambient_dim, ambient_dim))
print(
    f"\x1b[34mDistance to skew-symmetric matrices: {(np.linalg.norm(vec + vec.T)):.1e}\x1b[0m."
)

# Translate the orbit by an on-orthogonal matrix to make it non-orthogonal.
print("Now, translate the orbit non-orthogonally...")
inv = np.random.randn(ambient_dim, ambient_dim)  # Generically invertible.
pts = np.array([inv @ pt for pt in pts])
print_covariance_eigenvalues(pts)
print_norms(pts)

# Check whether orbit is orthogonal via skew-symmetricity of bottom eigenvector of Lie-PCA operator.
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=10,
    orbit_dim=1,
    method="PCA",
    correction=False,
    verbose=True,
)
eigvals, eigvecs = np.linalg.eig(lie_pca)
vec = eigvecs[:, np.argmin(eigvals)].reshape((ambient_dim, ambient_dim))
print(
    f"\x1b[34mDistance to skew-symmetric matrices: {(np.linalg.norm(vec + vec.T)):.1e}\x1b[0m."
)

# Orthonormalize the orbit.
print("Now, orthonormalize the orbit...")
pts, _ = orthonormalize(pts=pts)
print_covariance_eigenvalues(pts)
print_norms(pts)

# Check whether orbit is orthogonal via skew-symmetricity of bottom eigenvector of Lie-PCA operator.
lie_pca = get_lie_pca_operator(
    pts=pts,
    nb_neighbors=10,
    orbit_dim=1,
    method="PCA",
    correction=False,
    verbose=True,
)
eigvals, eigvecs = np.linalg.eig(lie_pca)
vec = eigvecs[:, np.argmin(eigvals)].reshape((ambient_dim, ambient_dim))
print(
    f"\x1b[34mDistance to skew-symmetric matrices: {(np.linalg.norm(vec + vec.T)):.1e}\x1b[0m."
)
