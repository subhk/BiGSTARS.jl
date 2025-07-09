# BiGSTARS.jl 

<!-- description --> 
  **Bi**-**G**lobal **St**ability **A**nalayis of **R**otating **S**tarified Flows (BiGSTARS :star: ): A linear stability analysis tool for Geophysical flows with Julia. 
<strong>https://subhk.github.io/BiGSTARSDocumentation/stable</strong>

 <!-- Badges -->
 <p align="left">
    <a href="https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml">
        <img alt="CI Status" src="https://github.com/subhk/BiGSTARS.jl/actions/workflows/CI.yml/badge.svg">
    </a>
    <a href="https://subhk.github.io/BiGSTARSDocumentation/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20release-blue">
    </a>
   <a href="https://subhk.github.io/BiGSTARSDocumentation/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-in%20development-orange">
    </a>
</p>



## Installation

Open the Julia REPL, press ] to enter **package-manager** mode, and then run the commands below.
This will add the package and automatically build (instantiate) all of its dependencies:

```julia
julia> ]
(@v1.11) pkg> add BiGSTARS
(@v1.11) pkg> instantiate
```

BiGSTARS.jl requires Julia v1.6 or later. However, the package has continuous integration testing on
Julia v1.10 (the current long-term release) and v1.11. 


## Examples

Example scripts can be found in the `examples/` directory. For the clearest overview, we recommend 
browsing them through the package’s documentation (work in progress! 😄). 
