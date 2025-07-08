# Installation instructions

You can install the latest version of `BiGSTARS.jl` using Julia’s built-in package manager. Just press `]` in the Julia REPL to enter package mode, then add the package and run `instantiate` to build all required dependencies.


```julia
julia> ]
(v1.11) pkg> add BiGSTARS
(v1.11) pkg> instantiate
```

We recommend installing `BiGSTARS.jl` using Julia’s built-in package manager, as this installs a stable, tagged release.  Later on, you can update `BiGSTARS.jl` to the latest tagged version again by using the package manager:


```julia
(v1.11) pkg> update BiGSTARS
```

**Note:** Some releases may introduce breaking changes to certain modules.  
If something stops working or your code behaves unexpectedly after an update, feel free to [open an issue](https://github.com/subhk/BiGSTARS.jl/issues) or [start a discussion](https://github.com/subhk/BiGSTARS.jl/discussions).  We're more than happy to help you get your model up and running again.


!!! warn "Julia 1.6 or newer required; Julia 1.10 or newer strongly encouraged"
    The latest version of `BiGSTARS.jl` requires **at least Julia v1.6** to run.  
    Installing `BiGSTARS.jl` with an older version of Julia will instead install the latest version that is compatible with your Julia installation.

    `BiGSTARS.jl` is continuously tested on **Julia v1.10** (the current long-term release) and **v1.11**.  
    _We strongly recommend using one of these tested Julia versions._
