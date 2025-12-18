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
date: 4 August 2025
bibliography: paper.bib
---


# Summary
`BiGSTARS.jl` is a software package written in the Julia programming language [@Bezanson2017] for investigating the stability of fluid flows relevant to the atmosphere and ocean. It performs linear stability analysis, a technique that determines whether small disturbances introduced into a flow will grow or decay over time. Unlike simpler one-dimensional approaches, bi-global analysis can handle flows whose properties vary in two spatial directions, enabling the study of realistic features such as fronts, jets, and vortices. The package is specifically designed for rotating and stratified fluids, capturing the physical processes essential to geophysical fluid dynamics.

# Statement of need
Linear stability analysis investigates the growth or decay of small perturbations about a basic state by linearizing the governing equations and solving the resulting eigenvalue problem [@drazin2004]. In geophysical fluid dynamics, the interplay between rotation (the Coriolis force) and stratification (the buoyancy force) gives rise to multiple instability mechanisms, including baroclinic instability driven by buoyancy flux and barotropic instability driven by horizontal shear [@pedlosky2013geophysical]. Such analyses are fundamental for understanding the onset of turbulence, the formation of eddies, and the associated energy transfer across scales in both oceanic and atmospheric systems.

The complexity and dimensionality of stability analyses have evolved substantially, giving rise to distinct methodological approaches with varying computational requirements [@theofilis2011global]. The classical one-dimensional (1D) stability analysis assumes that the basic state varies along only a single spatial direction. While computationally efficient, this approach presumes spatial homogeneity in the remaining directions and therefore often fails to capture the full dynamics of realistic geophysical flows. At the opposite extreme, the tri-global (3D global) stability analysis allows variations in all three spatial directions, providing the most complete representation of the underlying physics. However, this generality requires substantial computational resources, often beyond the reach of standard research computing infrastructures.

Bridging these two extremes, bi-global (2D global) linear stability analysis occupies an optimal middle ground between 1D and tri-global frameworks [@theofilis2011global]. In this approach, the basic state varies in two spatial directions while remaining homogeneous in the third, striking a balance between computational tractability and physical realism. Many geophysical flows naturally exhibit this structure, including atmospheric and oceanic jets [@pedlosky2013geophysical] as well as submesoscale oceanic fronts and filaments [@mcwilliams2016], where classical 1D analyses omit key dynamics and fully 3D approaches remain computationally prohibitive. 

To address this need, we present `BiGSTARS.jl`, a Julia‑based bi‑global stability solver designed to integrate seamlessly with the broader Julia scientific computing ecosystem. The package is distributed with validated benchmark examples and is intended to be readily accessible to the geophysical fluid dynamics community. `BiGSTARS.jl` implements a spectral–collocation framework in which the governing equations are discretized using Chebyshev polynomials in the vertical direction and Fourier modes in the horizontal direction, applied to a two‑dimensional basic state defined over a rectangular domain.



<!-- It is based on linearized Boussinesq equations of motion under the $f$-plane approximation by formulating a generalized eigenvalue problem, enforces boundary conditions via tau or penalty methods, and assembles a large, sparse generalized eigenvalue problem of the form $AX = λBX$, where $A$ and $B$ are the matrices, $X$ is the eigenvector and $\lambda$ is the eigenvalue. The package provides specialized features for the large eigenvalue problems characteristic of bi-global analysis. -->

<!-- - Shift-and-invert targeting: Efficiently find eigenvalues near specified regions of the complex plane, crucial for identifying for unstable modes.
- Multiple backend integration: Seamless switching between Arnoldi.jl, Arpack.jl, and KrylovKit.jl solvers with performance comparison tools. -->


## State of the field

Although several open-source packages exist for one-dimensional stability analyses (e.g., pyqg [@abernathey2022pyqg], eigentools [@eigentools_2021]), to the best of our knowledge, no fully documented open-source software currently offers a comprehensive bi-global eigenvalue solver capable of treating the linearized rotating Boussinesq equations of motion under the $f$‑plane approximation [@vallis2017atmospheric].


## Key features
BiGSTARS.jl leverages Chebyshev-Fourier discretization to handle vertically bounded and horizontally periodic domains — an optimal configuration for linear stability analyses of geophysical flows. The Chebyshev operators are constructed using Chebyshev–Gauss–Lobatto nodes, which are the extrema of first-kind Chebyshev polynomials  [@trefethen2000spectral]. Additionally, the framework offers flexible boundary condition handling, enabling the use of different types of mathematical boundary conditions (e.g., Dirichlet, Neumann) for each variable.

Additionally, the package is based on the shift-and-invert technique, which enables the efficient computation of eigenvalues in targeted regions of the complex plane, crucial for obtaining the most unstable modes. Users can seamlessly switch among multiple Julia eigen-solver backends — `ArnoldiMethod.jl` [@Stoppels], `Arpack.jl` [@shah2018], and `KrylovKit.jl` [@haegeman2025] — with built-in performance benchmarking tools. To address convergence challenges in large problems, `BiGSTARS.jl` employs adaptive convergence strategies, including automatic shift adjustments and retry logic, thus reducing the need for manual parameter tuning.

The package documentation includes a collection of validated examples that illustrate the key functionalities of the solver, such as setting up a basic state, specifying boundary conditions, and visualizing the results. These examples not only provide an accessible starting point for new users unfamiliar with bi-global stability analysis but also serve as reference cases for developing customized modules.

<!-- ## Mathematics

The package is designed to solve eigenvalue problem of the linearized Boussinesq equations of motion under the $f$-plane approximation,

$$\frac{D \mathbf{u}}{Dt}
    + \Big(v \frac{\partial U}{\partial y} + w \frac{\partial U}{\partial z} \Big) \hat{x}
    + f \hat{z} \times \mathbf{u} =
    -\nabla p + b \hat{z} + \nu \nabla^2 \mathbf{u}, \label{eq:1} \tag{1}$$

$$\frac{Db}{Dt} +  v \frac{\partial B}{\partial y} + w \frac{\partial B}{\partial z} 
    = \kappa \nabla^2 b \label{eq:2} \tag{2}$$

$$\nabla \cdot \mathbf{u} = 0, \label{eq:3} \tag{3}$$

where $\mathbf{u}\equiv(u,v,w)$ are the perturbation velocity in the $x$, $y$ and $z$-direction, respectively, 
$p$ is the perturbation pressure, $b=-g\rho/\rho_0$ ($\rho$ is the density perturbation relative to the reference density $\rho_0$, 
and $g$ is the gravitational acceleration). 
The variables $U(y,z)$ is mean flow in the $x$-direction and $B(y,z)$ is the buoyancy of basic state, 
which is in thermal-wind balance [@vallis2017atmospheric],
$$\frac{\partial U}{\partial z} = -\frac{\partial B}{\partial y} \label{eq:4} \tag{4}$$.  
For bi-global analysis, we assume normal mode solutions in the $x$-direction only,
$$[\mathbf{u},p,b](x,y,z,t) = \mathfrak{R}[\tilde{\mathbf{u}},\tilde{p},\tilde{b}](y,z) e^{ikx+\lambda t} \label{eq:5} \tag{5}$$, 
where $\mathfrak{R}$ denotes the real part, $k$ is the $x$-wavenumber and $\lambda$ is the complex frequency with real part describes the
growthrate and imaginary part denotes the frequency. The variables with superscript describes the eigenfunction. -->


# Acknowledgements

The author gratefully acknowledges Sohan Suresan for the stimulating discussions. The author acknowledges support from the Israel Science Foundation (ISF; grant 2054/23) and from the European Union through the European Research Council (ERC, 401 ML Transport 101163887) project. Views and opinions expressed are, however, those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

# References
