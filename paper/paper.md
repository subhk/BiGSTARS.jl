---
title: 'BiGSTARS.jl: A Julia package for bi-global stability analysis for rotating stratified flows'
tags:
  - Julia
  - spectral methods
  - instability
  - geophysical fluid dynamics
  - bi-global analysis
  - eigenvalue problems
authors:
  - name: Subhajit Kar
    orcid: 0000-0001-9737-3345
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
  - name: Tel Aviv University, Israel
    index: 1
date: 31 July 2025
bibliography: paper.bib
---


# Summary
`BiGSTARS.jl` a Julia-based [@Bezanson2017] package developed for performing bi-global linear stability analysis for rotating stratified atmospheric and oceanic flows.


# Statement of need
Linear stability analysis is fundamental to understanding the dynamics of geophysical flows, from the onset of turbulence in oceanic currents to the formation of atmospheric waves and instabilities. The complexity and dimensionality of these analyses have evolved significantly, leading to distinct approaches with different computational requirements and physical insights.
One-Dimensional (1D) Stability Analysis represents the classical approach where the base flow varies in only one spatial direction, typically the wall-normal direction in boundary layers or the vertical direction in stratified flows. Tri-Global (3D Global) Stability Analysis represents the most general case where the base flow varies in all three spatial directions, requiring full three-dimensional eigenvalue problems. 

Bi-global linear stability analysis sits at the `Goldilocks' point between the 1D and Tri-Global stability analyses [@theo]. 
Geophysical fluid dynamics frequently encounters flows that are inherently bi-global in nature, where classical 1D analysis fails to capture essential physics:

Atmospheric Phenomena:
- Mountain Wave Dynamics: Airflow over mountain ranges creates lee waves and rotors that vary significantly in both the vertical and cross-mountain directions. 1D analysis assuming only vertical variation misses critical three-dimensional wave interactions and breaking mechanisms.
- Mesoscale Convective Systems: Organized convection in the atmosphere often develops two-dimensional structure in the vertical plane, with updrafts and downdrafts that cannot be captured by simple vertical profiles.
- Jet Stream Instabilities: Upper-level atmospheric jets exhibit both vertical shear and horizontal curvature, requiring bi-global analysis to understand their meandering and breakdown.

Oceanic Applications:
- Western Boundary Current Separation: The Gulf Stream and Kuroshio Current exhibit complex separation dynamics where the flow varies significantly in both the vertical and cross-stream directions.
- Overflow Dynamics: Dense water flowing over topographic features (Denmark Strait, Mediterranean outflow) creates flows that are fundamentally two-dimensional in the plane perpendicular to the flow direction.
- Coastal Upwelling Systems: Wind-driven upwelling creates complex velocity and density structures that vary both vertically and across the shelf.

At the same time, it makes the eigenvalue problem orders of magnitude smaller than a full 3D global stability solver. This economy lets researchers map growth rates and mode structures of different instability processes on standard workstations rather than supercomputers.

This statement motivates the development of `BiGSTARS.jl` - an open‑source, Julia‑based bi‑global solver that integrates with the wider Julia scientific ecosystem, ships with validated examples, and accessible to the wider geophysical fluid dynamics community.

`BiGSTARS.jl` implements a bi-global spectral–collocation method tailored for stability analysis of geophysical flows. The code defines a two-dimensional base flow over a rectangular domain and discretizes it using Chebyshev polynomials in the vertical direction and Fourier modes in horizontal direction. It is based on linearized Boussinesq equations of motion under the $f$-plane approximation by formulating a generalized eigenvalue problem, enforces boundary conditions via tau or penalty methods, and assembles a large, sparse generalized eigenvalue problem of the form $AX = \lambda BX$, where $A$ and $B$ are the matrices, $X$ is the eigenvector and $\lambda$ is the eigenvalue. The package provides specialized features for the large eigenvalue problems characteristic of bi-global analysis:
- Shift-and-invert targeting: Efficiently find eigenvalues near specified regions of the complex plane, crucial for identifying for unstable modes.
- Multiple backend integration: Seamless switching between Arnoldi.jl, Arpack.jl, and KrylovKit.jl solvers with performance comparison tools.

Documented examples appear in the package's documentation, providing a starting point for new users and for the development of new or customized modules. 

## State of the field

To the author’s best knowledge, there is currently no freely available, fully documented open-source package that delivers a turnkey bi-global eigenvalue solver capable of treating the lineralized Boussinesq-equation dynamics of rotating, stratified geophysical flows.

## Mathematics

The package is designed to solve eigenvaue problem of the linearized Boussinesq equations of motion, 

$$\frac{D {u}}{Dt}
    + \Big(v \frac{\partial U}{\partial y} + w \frac{\partial U}{\partial z} \Big) \hat{x}
    + \hat{z} \times \mathbf{u} =
    -\nabla p + b \hat{z} + \nu \nabla^2 \mathbf{u}, \label{eq:1} \tag{1}$$

$$\frac{Db}{Dt}
    +  v \frac{\partial B}{\partial y} + w \frac{\partial B}{\partial z} 
    = \kappa \nabla^2 b \label{eq:2} \tag{2}$$

$$\nabla \cdot \mathbf{u} = 0, \label{eq:3} \tag{3}$$


# Acknowledgements

# References
