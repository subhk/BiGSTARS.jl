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

### Distributed sparse solve

[`solve`](@ref) ships each assembled pencil to PETSc as a distributed sparse
`MatMPIAIJ` and factorizes it with a parallel direct solver (`mat_solver`, default
`:mumps`) for the shift-and-invert inner solves — there is no separate dense path.
The banded / block sparsity above is what keeps that factorization cheap.

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

[`solve`](@ref) keyword arguments control the shift-and-invert solve:

- **`nev`**, **`which`**, **`tol`**, **`maxiter`** — number of eigenvalues, target region
  (`:LM` nearest σ, `:LR`, `:SR`, …), tolerance, and the SLEPc iteration cap.
- **`n_tries`**, **`Δσ₀`**, **`incre`**, **`ϵ`** — the adaptive-σ retry budget: how many
  shifts to try around `sigma_0`, their spacing / growth, and the successive-eigenvalue
  tolerance that ends the loop.
- **`mat_solver`** (default `:mumps`) — parallel direct solver for the inner factorization
  (`:mumps`, `:superlu_dist`, `:petsc`); **`eps_type`** (default `:krylovschur`) the SLEPc EPS.

Results come back sorted nearest the shift (`results[i].eigenvalues[1]` is the mode at `σ`);
reorder with `sort_evals(λ, Χ, :R)` for the largest growth rate.

```julia
solve(cache, k_values; sigma_0 = 0.5, nev = 3, which = :LR, mat_solver = :mumps)
```

## Time to first solve

BiGSTARS ships a `PrecompileTools` workload that exercises `discretize → assemble` on a tiny
problem at build time, so the first real call in a fresh session avoids most of the
discretization / assembly compilation latency. (The eigensolve lives in the SLEPc/PETSc
extension and is not precompiled here.)
