# <img src="./gfd_instability.svg" height="80" width="90">  BiGSTARS.jl 

<!-- description --> 
  **Bi**-**G**lobal **St**ability **A**nalysis of **R**otating **S**tratified Flows (BiGSTARS): A linear stability analysis tool for geophysical flows with Julia. 

Bi-global stability analysis offers a pragmatic alternative between 1D (too idealized) and fully tri-global (often too expensive) approaches. BiGSTARS.jl gives geophysical fluid dynamicists a practical middle ground with a symbolic equation DSL, ultraspherical spectral discretizations, and shift-and-invert eigensolvers — so you can resolve key instabilities without massive computational resources.

Write your equations — the code does the rest.

```julia
prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)
@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))
@equation sigma * Lap(psi) = U * dx(Lap(psi)) - E * Lap(Lap(psi))
@bc left(psi) = 0
@bc right(psi) = 0
cache = discretize(prob)
results = solve(cache, k_values; sigma_0=0.02)
```

### Key Features

- **Symbolic equation DSL** — Write PDEs with `dx`, `dy`, `dz`; automatic discretization into sparse GEVP matrices
- **Ultraspherical spectral method** — Fully sparse Chebyshev operators (Olver & Townsend, 2013); Fourier in coefficient space
- **Generalized BCs** — Dirichlet, Neumann, Robin, higher-order, coupled, and eigenvalue-dependent boundary conditions
- **Derived variables** (`@derive`) — Eliminate auxiliary fields via inverse operators; reduces system dimension
- **Wavenumber-separated caching** — Discretize once, assemble cheaply per wavenumber; parallel sweeps with `parallel=true`
- **2D background fields** — Full support for fields varying in both Fourier and Chebyshev directions
- **Post-processing** (`@compute`) — Evaluate any expression on eigenvectors using the same DSL syntax
- **Multiple solvers** — Arnoldi, ARPACK, KrylovKit with adaptive shift-and-invert

 <!-- Badges -->
 <p align="left">
    <a href="https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml">
        <img alt="CI Status" src="https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml/badge.svg">
    </a>
</p>

## Docs
<!-- Badges -->
 <p align="left">
    <a href="https://subhk.github.io/BiGSTARSDocumentation/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20-blue">
    </a>
      <a href="https://subhk.github.io/BiGSTARSDocumentation/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-dev%20-orange">
    </a>
    <a href="https://doi.org/10.5281/zenodo.18385010">
        <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.18385010.svg">
    </a>
    <a style="border-width:0" href="https://doi.org/10.21105/joss.09368">
         <img src="https://joss.theoj.org/papers/10.21105/joss.09368/status.svg" alt="DOI badge" >
    </a>
</p>


## Installation

Open the Julia REPL, press ] to enter **package-manager** mode, and run the following commands. 
These will add **BiGSTARS** and automatically instantiate all of its dependencies:

```julia
julia> ]
(@v1.11) pkg> add BiGSTARS
(@v1.11) pkg> instantiate
```

BiGSTARS.jl requires **Julia 1.10** or newer.


## Examples

Example scripts can be found in the `examples/` directory. For the clearest overview, we recommend 
browsing them through the package’s documentation.


## Contributing

If you’re interested in contributing to the development of ``BiGSTARS.jl``, we’re excited to have your help—no matter 
how big or small the contribution. New perspectives are especially valuable: fresh eyes on the code often 
reveal issues or improvements that existing developers may have missed.

For more information, check out our [contributors' guide](https://github.com/subhk/BiGSTARS.jl?tab=contributing-ov-file)
