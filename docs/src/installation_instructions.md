# Installation instructions

You can install the latest version of `BiGSTARS.jl` using Julia’s built-in package manager. 
Just press `]` in the Julia REPL to enter package mode, then add the package and run instantiate 
to build all required dependencies.


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
If something stops working or your code behaves unexpectedly after an update, feel free to [open an issue](https://github.com/subhk/BiGSTARS.jl/issues).  
We're more than happy to help you get your model up and running again.


!!! warn "Julia 1.10 or newer required"
    BiGSTARS.jl is continuously tested on Julia v1.10 (the current long-term support), v1.11 and v1.12. For optimal stability and reproducibility, we strongly recommend using one of these tested Julia versions.

## Solving requires PETSc/SLEPc

`]add BiGSTARS` installs and loads everywhere — the discretization / DSL pipeline
is pure Julia. **Solving** eigenproblems, however, goes through SLEPc over PETSc
(the only eigensolver), which needs:

- A **complex-scalar** system build of PETSc and SLEPc
  (`./configure --with-scalar-type=complex`), with `PETSC_DIR`, `PETSC_ARCH`, and
  `SLEPC_DIR` exported.
- The Julia packages `MPI`, `PetscWrap`, `SlepcWrap`, with MPI.jl bound to the
  same MPI used to build PETSc. Importing all three activates the solver.

Without this backend, [`solve`](@ref) raises an install hint. See
[Distributed (MPI)](mpi.md) for the full environment setup, and run with
`mpiexec -n P julia --project=. script.jl`.

