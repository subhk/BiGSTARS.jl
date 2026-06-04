# API Reference

This page lists the public API exported by `BiGSTARS`. Most workflows use the DSL macros plus the `discretize` / `solve` path:

```julia
using BiGSTARS

cache = discretize(prob)
results = solve(cache, k_values; sigma_0=0.02)
```

`solve` is the SLEPc/PETSc eigensolver, provided by the `BiGSTARSMPIExt` extension
(import `MPI`, `PetscWrap`, `SlepcWrap`). For lower-level control, assemble matrices
with `assemble` and drive SLEPc yourself.

## Public API

```@autodocs
Modules = [BiGSTARS]
Private = false
Order = [:type, :macro, :function]
```
