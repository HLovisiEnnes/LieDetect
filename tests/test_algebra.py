from algebra import (
    get_random_lattice,
    get_lattices,
    get_constrained_partitions,
    are_representations_equivalent,
    get_canonical_pushforward_algebra,
)


" Compare a random lattice to the whole list of lattices "
print("\nCompare a random lattice to the whole list of lattices")

# Parameters.
lattice_rank, ambient_rank, frequency_max = 2, 3, 4
# lattice_rank, ambient_rank, frequency_max = 3, 4, 2

# Generate a random lattice.
lattice = get_random_lattice(
    lattice_rank=lattice_rank, ambient_rank=ambient_rank, frequency_max=frequency_max
)
print("Random lattice:", lattice)

# Generate all lattices, up to span-equivalence, not max dim orbit.
lattices = get_lattices(
    lattice_rank=lattice_rank,
    ambient_rank=ambient_rank,
    frequency_max=frequency_max,
    method="span-equivalence",
    span_ambient_space=False,
)
print("Nb of lattices (up to span-equivalence, not max dim orbit):", len(lattices))

# Generate all lattices, up to orbit-equivalence, max dim orbit.
lattices = get_lattices(
    lattice_rank=lattice_rank,
    ambient_rank=ambient_rank,
    frequency_max=frequency_max,
    method="span-equivalence",
    span_ambient_space=True,
)
print("Nb of lattices (up to span-equivalence, max dim orbit):", len(lattices))

# Generate all lattices, up to orbit-equivalence, not max dim orbit.
lattices = get_lattices(
    lattice_rank=lattice_rank,
    ambient_rank=ambient_rank,
    frequency_max=frequency_max,
    method="orbit-equivalence",
    span_ambient_space=False,
)
print("Nb of lattices (up to orbit-equivalence, not max dim orbit):", len(lattices))

# Generate all lattices, up to orbit-equivalence, max dim orbit.
lattices = get_lattices(
    lattice_rank=lattice_rank,
    ambient_rank=ambient_rank,
    frequency_max=frequency_max,
    method="orbit-equivalence",
    span_ambient_space=True,
)
print("Nb of lattices (up to orbit-equivalence, max dim orbit):", len(lattices))

# Compare lattices.
equal_lattices = [
    are_representations_equivalent(group="torus", rep0=lattice, rep1=ltc)
    for ltc in lattices
]
print("Nb of lattices equivalent to the given one:", sum(equal_lattices))
idx = next((i for i, eq in enumerate(equal_lattices) if eq))
are_representations_equivalent(
    group="torus", rep0=lattice, rep1=lattices[idx], verbose=True
)


" Maximal frequency must be large enough "
print("\nMaximal frequency must be large enough")


lattice_rank, ambient_rank, frequency_max = 2, 5, 1
try:
    lattices = get_lattices(
        lattice_rank=lattice_rank,
        ambient_rank=ambient_rank,
        frequency_max=frequency_max,
        method="orbit-equivalence",
    )
except Exception as e:
    print(e)


" Partitions "
print("\nPartitions")


n = 10

# For SU(2).
group = "SU(2)"
partitions = get_constrained_partitions(
    group=group, ambient_dim=n, span_ambient_space=True
)
print("Partitions:", partitions)

# For SO(3).
group = "SO(3)"
partitions = get_constrained_partitions(
    group=group, ambient_dim=n, span_ambient_space=True
)
print("Partitions:", partitions)


" Pushforward algebras "
print("\nPushforward algebras")


# For the torus.
lattice = ((1, 2, 1), (0, 1, 2))
group = "torus"
pfwd_alg = get_canonical_pushforward_algebra(group=group, rep_type=lattice)
print("Pushforward algebra:", *[mat for mat in pfwd_alg])

# For SU(2).
partition = (3, 4)
group = "SU(2)"
pfwd_alg = get_canonical_pushforward_algebra(group=group, rep_type=partition)
print("Pushforward algebra:", *[mat for mat in pfwd_alg])
