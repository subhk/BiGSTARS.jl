# Performance & Sparsity

BiGSTARS assembles **sparse** generalized eigenvalue problems and solves them with
shift-and-invert. This page collects the knobs that control speed and sparsity.

## Sparsity is automatic

The discretized operators are sparse by construction:

- **Chebyshev (`z`) derivatives** — banded ultraspherical operators (Olver–Townsend).
- **Fourier (`y`) derivatives** — diagonal in coefficient space (`d/dy → ikₘ`).
- **Variable coefficients** `f(z)·…` — banded multiplication operator in the ultraspherical
  basis, with bandwidth equal to the number of significant Chebyshev modes of `f`. Density
  falls like `band/N`, so well-resolved problems stay sparse.
- **Constant-coefficient, multi-dimensional** problems decouple by Fourier mode → block
  structure, often `< 1 %` fill.

The only genuinely dense case is a coefficient field that truly needs `O(N)` spectral modes
(very steep / non-smooth relative to the resolution) — there the operator *is* dense, and
dense storage is the right choice.

### Sparse vs dense solve path

[`solve`](@ref) estimates the assembled fill and automatically routes to a sparse
(`SparseMatrixCSC` + sparse LU) or dense path. Override with the `sparse` keyword:

```julia
solve(cache, k_values; sigma_0 = 0.02, sparse = true)   # force sparse
solve(cache, k_values; sigma_0 = 0.02, sparse = false)  # force dense
```

## Derived variables (`@derive`)

By default, derived variables are **augmented** into the system: a derived `v` defined by
`Op(v) = rhs` becomes a regular unknown plus the sparse constraint equation `Op(v) − rhs = 0`,
instead of forming the dense inverse `Op⁻¹`. This keeps the operator sparse and even handles
operators whose inverse the elimination path cannot build.

```julia
cache = discretize(prob)                        # augment_derived = true (default)
cache = discretize(prob; augment_derived = false)  # legacy: eliminate v via Op⁻¹ (denser)
```

The augmented system has a singular `B` (the constraint rows carry no eigenvalue), which
produces spurious/infinite eigenvalues. `solve` filters them, **but you should still
shift-target the physical region** (set `sigma_0` near the mode of interest) rather than
asking for the smallest-magnitude eigenvalue.

## Eigensolver tuning

[`SolverConfig`](@ref) controls the shift-and-invert solve:

- **`krylovdim`** (default `30`) — Krylov subspace dimension for `method = :Krylov`. The cost
  scales steeply with this; the default is sized for `nev = 1`–few. Raise it only for hard /
  clustered spectra. It is clamped at runtime to `[nev+2, n]`.
- **`sortby`** (default `:nearest`) — order of the returned eigenvalues. `:nearest` puts the
  mode closest to the shift `σ₀` first (what shift-and-invert targets). Use `:R` for the
  largest growth rate, `:I`, or `:M` (magnitude).
- **`nev`**, **`which`**, **`tol`**, **`maxiter`**, **`n_tries`** — number of eigenvalues,
  target region, tolerance, iteration cap, and adaptive-shift retry budget.

```julia
solve(cache, k_values; sigma_0 = 0.5, method = :Krylov, nev = 3, krylovdim = 60, sortby = :R)
```

## Time to first solve

BiGSTARS ships a `PrecompileTools` workload that exercises `discretize → assemble → solve` on
a tiny problem at build time, so the first real call in a fresh session avoids most of the
compilation latency.
