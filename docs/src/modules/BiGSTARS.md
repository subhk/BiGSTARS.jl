# API Reference

This page lists the public API exported by `BiGSTARS`. Most workflows use the DSL macros plus the `discretize` / `solve` path:

```julia
using BiGSTARS

cache = discretize(prob)
results = solve(cache, k_values; sigma_0=0.02)
```

For lower-level control, assemble matrices with `assemble` and pass them to `EigenSolver`.

## Public API

```@autodocs
Modules = [BiGSTARS]
Private = false
Order = [:type, :macro, :function]
```
