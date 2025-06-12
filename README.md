# LieDetect

## Background

In the preprint https://arxiv.org/abs/2309.03086, we suggested a new algorithm to estimate representations of compact Lie groups from finite samples of their orbits. The algorithm takes as an input a point cloud $X$ in $\mathbb{R}^n$, a compact Lie group $G$ (among $\mathrm{SO}(2)$, $T^d$ for $d\geq 1$, $\mathrm{SO}(3)$, and $SU(2)$), and returns a representation of $G$ in $\mathbb{R}^n$ and an orbit $\mathcal{O}$ that, if succeeded, lies close to $X$.

At its core, the algorithm enumerates the representations of the group in $\mathbb{R}^n$ (up to orbit-equivalence), an determines the best one, at the level of Lie algebras, and through an optimization over the orthogonal group $\mathrm{O}(n)$. The following animation illustrates the case of a representation of $\mathrm{SO}(2)$ in $\mathbb{R}^{12}$. Additional illustrations are found at https://www.youtube.com/playlist?list=PL_FkltNTtklBQlwrGyAnisJ-lGiLFeTVw.

![til](./Animations/optimization_circle.gif)

All the illustrations found in our paper are implemented in *LieDetect_Illustrations.ipynb*. In addition, this repo contains several other notebooks, described below.

## Tutorial

The Jupyter notebook *LieDetect_Tutorial.ipynb* contains several basic experiments, displaying the possibilities of $\texttt{LieDetect}$.

## Image analysis

The notebook *LieDetect_Experiments_ImageAnalysis.ipynb* gathers our experiments with 2D and 3D images. As it turns out, common image transformations, such as translations and rotations, when embedded in the Euclidean space, form an orbit of a representation of a Lie group. This information can subsequently be used for various Machine Learning tasks.

![til](./Animations/gorillas_circle.gif)

## Physics

In *LieDetect_Experiments_Physics.ipynb*, we put $\texttt{LieDetect}$ in practice on two classical mechanics systems: the three-body problem, and the multidimensional harmonic oscillator. Suprisingly, we found that certain Broucke periodic orbits are very well described by representation orbits of $\mathrm{SO}(2)$.

![til](./Animations/Broucke_Orbit_A2.gif)

## Chemistry

Our last experiment is found in *LieDetect_Experiments_Cyclooctane.ipynb*. As found recently, the space of conformers of cyclooctane exhibits an intriguing topology: that of the union between a sphere and a Klein bottle. The Klein bottle does not support an action of the torus $T^2$, but it has an action of $\mathrm{SO}(2)$, though not transitive. We find such a representation, after embedding the transformers in the Euclidean space.

![til](./Animations/cyclooctane_circle.gif)
