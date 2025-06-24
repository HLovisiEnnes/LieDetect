# Third-party imports.
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib

# Local imports.
from algebra import get_random_lattice, get_canonical_pushforward_algebra
from orbits import sample_orbit_from_algebra, sample_orbit_from_group

matplotlib.use("TkAgg")


" Sample random orbit - SO(2) "
print("\nSample random orbit - SO(2)")

# Generate orbit.
group = "torus"
ambient_dim = 8
nb_points = 1000
frequency_max = 10
group_dim = 1
method = "random"
orbit, rep_type = sample_orbit_from_group(
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

# Plot.
orbit_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(orbit)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax.scatter(orbit_pca[:, 0], orbit_pca[:, 1], orbit_pca[:, 2], c="black")
plt.title("Orbit of SO(2)")
plt.show()


" Sample random orbit - T^2 "
print("\nSample random orbit - T^2")

# Generate orbit.
group = "torus"
ambient_dim = 6
nb_points = 20000
frequency_max = 2
group_dim = 2
method = "random"
orbit, _ = sample_orbit_from_group(
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

# Plot.
orbit_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(orbit)
_, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax.scatter(orbit_pca[:, 0], orbit_pca[:, 1], orbit_pca[:, 2], c="black", s=3)
plt.title("Orbit of T^2")
plt.show()


" Sample random orbit - SU(2) "
print("\nSample random orbit - SU(2)")


# Generate orbit.
group = "SU(2)"
ambient_dim = 7
nb_points = 50000
method = "uniform"
orbit, _ = sample_orbit_from_group(
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

# Plot.
orbit_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(orbit)
_, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax.scatter(orbit_pca[:, 0], orbit_pca[:, 1], orbit_pca[:, 2], c="black", s=2)
plt.title("Orbit of SU(2)")
plt.show()


" Sample a given representation - T^3 "
print("\nSample a given representation - T^3")


# Generate a representation type (lattice).
lattice_rank = 3
ambient_rank = 4
frequency_max = 3
lattice = get_random_lattice(
    lattice_rank=lattice_rank,
    ambient_rank=ambient_rank,
    frequency_max=frequency_max,
)

# Get corresponding pushforward algebra.
pfwd_alg = get_canonical_pushforward_algebra(group="torus", rep_type=lattice)

# Generate orbit from random point.
orbit = sample_orbit_from_algebra(
    group="torus",
    rep_type=lattice,
    algebra=pfwd_alg,
    x=np.random.rand(2 * ambient_rank),
    nb_points=40000,
    method="random",
    verbose=True,
)

# Plot.
orbit_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(orbit)
_, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax.scatter(orbit_pca[:, 0], orbit_pca[:, 1], orbit_pca[:, 2], c="black", s=2)
plt.title("Orbit of T^3, random initial point")
plt.show()

# Generate orbit from "homogeneous" point.
orbit = sample_orbit_from_algebra(
    group="torus",
    rep_type=lattice,
    algebra=pfwd_alg,
    x=np.ones(2 * ambient_rank),
    nb_points=40000,
    method="random",
    verbose=True,
)

# Plot.
orbit_pca = sklearn.decomposition.PCA(n_components=3).fit_transform(orbit)
_, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
ax.scatter(orbit_pca[:, 0], orbit_pca[:, 1], orbit_pca[:, 2], c="black", s=2)
plt.title("Orbit of T^3, homogeneous initial point")
plt.show()
