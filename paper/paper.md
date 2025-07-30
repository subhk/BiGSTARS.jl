---
title: 'BiGSTARS.jl: A Julia package for bi-global stability analysis for rotating stratified flows'
tags:
  - Julia
  - instability
  - geophysical fluid dynamics
  - bi-global
  - spectral method
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
Bi-global linear stability analysis sits at the `Goldilocks' point between traditional local (or, 1D) and fully 3D (or tri-global) stability analyses [@theo]. By allowing the base flow to vary in two horizontal directions, it resolves key structural features in the geophysical flows - meanders, fronts, eddies, and coastal boundaries that local stability analysis fails to capture. At the same time, it makes the eigenvalue problem orders of magnitude smaller than a full 3D global stability solver. This economy lets researchers map growth rates and mode structures of different instability processes on standard workstations rather than supercomputers.

This statement motivates the development of `BiGSTARS.jl` - an open‑source, Julia‑based bi‑global solver that integrates with the wider Julia scientific ecosystem, ships with validated examples, and accessible to the wider geophysical fluid dynamics community.

`BiGSTARS.jl` implements a bi-global spectral–collocation method tailored for stability analysis of geophysical flows. The code defines a two-dimensional base flow over a rectangular domain and discretizes it using Chebyshev polynomials in the vertical direction and Fourier modes in horizontal direction. It is based on linearized Boussinesq equations of motion under the $f$-plane approximation by formulating a generalized eigenvalue problem, enforces boundary conditions via tau or penalty methods, and assembles a large, sparse generalized eigenvalue problem of the form $AX = \lambda BX$, where $A$ and $B$ are the matrices, $X$ is the eigenvector and $\lambda$ is the eigenvalue. The package relies on Julia’s rich linear algebra ecosystem for computing the most unstable eigenmodes.

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

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
