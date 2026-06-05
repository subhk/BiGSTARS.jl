# Design: Phase 2b-ii — distributed spurious-mass filter (all ranks restricted)

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Builds on:** 2a (`assemble_rows`, distributed `B`), 2b-i (`discretize_distributed`, `row_range`).
**Completes:** Phase 2 (full distributed discretize+assemble).

## Goal

Make the singular-`B` spurious-mode filter fully distributed so **no rank needs a
full `B`** — letting `discretize_distributed` restrict **every** rank (drop the
"group roots keep the full cache" carve-out from 2b-i). True extreme-N: even one
node cannot hold the full components.

## Why this is the last blocker

2b-i restricts non-root ranks but keeps the **full cache on each group root**,
solely because the mass filter calls `_assemble_B_full(cache, k)` (needs full `B`)
on the root. Removing that dependency removes the only reason a root holds full.

## Feasibility (verified against installed PetscWrap 0.1.5)

- `MatMult(mat, x, y)` — wrapped (`mat.jl:196`).
- `VecDuplicate(vec)` — wrapped (`vec.jl:173`).
- `VecGetArray`/`VecRestoreArray` — wrapped (already used in `_gather_eigenpairs`).
- `VecNorm` — **NOT** wrapped. Not needed: the norm is `sqrt(MPI.Allreduce(Σ|local|²))`
  using the local slice from `VecGetArray`. No new wrappers.

The eigenvector is a distributed `Vec` (`vr`) *before* it is gathered, and the
distributed `B` is already built — so the mass is computed on the distributed
objects in hand.

## Components

### A. `_keep_by_mass(masses; rtol=1e-6)` — pure (src/results.jl)
The keep-rule extracted from `_filter_physical_modes`, operating on precomputed
masses instead of `(Χ, B)`:
```julia
function _keep_by_mass(masses::AbstractVector{<:Real}; rtol::Float64=1e-6)
    isempty(masses) && return Int[]
    scale = maximum(masses)
    scale == 0.0 && return collect(eachindex(masses))     # degenerate → keep all
    keep = findall(>(rtol * scale), masses)
    isempty(keep) && return collect(eachindex(masses))    # fallback → keep all
    return keep
end
```
Unit-tested cross-platform (drops near-zero masses; all-zero / empty fallbacks).
`_filter_physical_modes` stays (used by the non-distributed/full path) — unchanged.

### B. Distributed mass in `_gather_eigenpairs` (ext/BiGSTARSMPIExt.jl)
Thread the distributed `B` Mat in; compute one mass per converged eigenpair while
its `Vec` `vr` is in hand. Signature becomes
`_gather_eigenpairs(eps, A, B, nconv, N, comm) -> (λ, Χ, masses)`:
```julia
Bvr = VecDuplicate(vr)
for ie in 0:(nconv-1):
    vpr, _, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
    local_arr, ref = VecGetArray(vr)
    nv2 = sum(abs2, local_arr)                      # ‖vr‖² local
    sendbuf = Vector{ComplexF64}(local_arr)
    VecRestoreArray(vr, ref)
    MatMult(B, vr, Bvr)
    bl, bref = VecGetArray(Bvr)
    nb2 = sum(abs2, bl)                             # ‖Bvr‖² local
    VecRestoreArray(Bvr, bref)
    nv2g = MPI.Allreduce(nv2, +, comm); nb2g = MPI.Allreduce(nb2, +, comm)  # collective
    masses[ie+1] = sqrt(nb2g) / sqrt(max(nv2g, eps()))
    # gather sendbuf to rank 0 as before (λ on rank 0; masses replicated on all ranks)
end
VecDestroy(Bvr)
```
`masses` is computed on every rank (Allreduce-replicated). Eigenvalues/vectors
gathered to rank 0 exactly as today. `MatMult`/`Allreduce`/the extra `VecGetArray`
are collective and run on every rank identically (collective-safe).

### C. Filter in `_solve_one_adaptive` (ext)
Replace the rank-0 `_filter_physical_modes(λ, Χ, _assemble_B_full(cache, k))` with
the distributed masses:
```julia
keep = _keep_by_mass(masses)          # masses from _gather_eigenpairs (replicated)
λf = λ[keep]; Χf = Χ[:, keep]
λf, Χf = sort_eigenvalues!(λf, Χf, :nearest; σ=σ)
```
`_assemble_B_full` is no longer called in the solve path (kept defined; now unused
there). The control flow (rank-0 decision, `MPI.bcast(stop)`) is unchanged.

### D. `discretize_distributed` (ext): restrict ALL ranks
Drop the carve-out — always restrict:
```julia
return restrict_cache_rows(cache, rstart, rend)   # every rank, including group roots
```

## Data flow / collective-safety
- `MatMult(B, vr, Bvr)` + the two `MPI.Allreduce`s run on every rank for each
  eigenpair (same count = `nconv`, replicated globally) → collective-safe, no new
  divergence. Masses are identical on all ranks; the keep decision (rank 0) is thus
  consistent with what every rank computed.
- No rank builds a full `B` anymore; `_assemble_B_full` drops out of the hot path.
- One extra `Vec` (`Bvr` via `VecDuplicate`) per gather, destroyed after the loop.

## Testing
- **Cross-platform (pure):** `_keep_by_mass` — drops a near-zero mass among `O(1)`
  ones; all-zero → keep all; empty → empty; single mode kept.
- **MPI CI (`mpi.yml`):** the existing singular-`B` spurious test (augmented
  descriptor, physical `σ=-1/π²`) must still drop the infinite modes and keep the
  physical one — now via distributed masses; and a `discretize_distributed`
  (all-ranks-restricted) solve must match the analytic reference at n=1/2/4.

## Risks
- Touches the v4.0.0/2a-verified `_gather_eigenpairs`/filter. The mass is the same
  formula (`‖Bχ‖/‖χ‖`) as `_filter_physical_modes`, just distributed — CI confirms
  equivalence via the spurious test.
- `MatMult`/`Allreduce`/`VecDuplicate` path is CI-only verifiable.
- Slight per-eigenpair cost (one extra distributed mat–vec) — negligible vs the
  solve.

## Out of scope
- Nothing further — 2b-ii completes the distributed discretize+assemble effort.
  (Reducing *peak* discretize memory still requires matrix-free operators, a
  separate architecture, as established.)
