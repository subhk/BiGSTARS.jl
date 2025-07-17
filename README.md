# BiGSTARS.jl 

<!-- description --> 
  **Bi**-**G**lobal **St**ability **A**nalysis of **R**otating **S**tratified Flows (BiGSTARS :star: ): A linear stability analysis tool for Geophysical flows with Julia. 

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
</p>

## Installation

Open the Julia REPL, press ] to enter **package-manager** mode, and run the following commands. 
These will add **BiGSTARS** and automatically instantiate all of its dependencies:

```julia
julia> ]
(@v1.11) pkg> add BiGSTARS
(@v1.11) pkg> instantiate
```

BiGSTARS.jl requires **Julia 1.6** or newer. Continuous-integration tests currently run on **Julia 1.10** (the current long-term support) and **Julia 1.11**.


## Examples

Example scripts can be found in the `examples/` directory. For the clearest overview, we recommend 
browsing them through the package’s documentation. 
