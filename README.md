# BiGSTARS.jl 

[![Build Status](https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml?query=branch%3Amain)

<!-- description -->
<p>
  <strong> A linear stability analysis tool for Geophysical flows with Julia. 
    https://fourierflows.github.io/GeophysicalFlowsDocumentation/stable</strong>
</p>


## Installation

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to instantiate/build all the required dependencies

```julia
julia> ]
(@v1.11) pkg> add BiGSTARS
(@v1.11) pkg> instantiate
```

BiGSTARS.jl requires Julia v1.6 or later. However, the package has continuous integration testing on
Julia v1.10 (the current long-term release) and v1.11. 


## Examples

See `examples/` for example scripts. These examples are best viewed by browsing them within 
the package's [documentation]. 
