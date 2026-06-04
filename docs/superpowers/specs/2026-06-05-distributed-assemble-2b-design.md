# Design: Phase 2b-i — MPI-coupled `discretize_distributed` (slice-on-store)

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Builds on:** Phase 2a (`assemble_rows`, `_build_petsc_mats_local`).
**Supersedes:** the earlier "hidden-in-solve" 2b sketch (flawed — gave no memory
benefit because the caller retained the full cache).

## Goal

Cut **steady-state per-rank component memory** by having the user hold a
per-rank **row-restricted** cache instead of the full one. A new
`discretize_distributed(prob; ngroups)` (extension) builds the full cache, computes
each rank's owned rows for its group's pencil, and returns a cache restricted to
those rows (full GC'd) on non-root ranks. `solve(cache; ngroups)` consumes it.

## Honest scope (reaffirmed)

- **Peak memory during `discretize_distributed` is NOT reduced** (the full cache is
  built first; products/inverses force full intermediates).
- **Group roots keep the FULL cache** (needed for the singular-B mass filter), so
  this does NOT yet help the extreme-N case where even one node can't hold the full
  components. That is **Phase 2b-ii** (distributed filter → all ranks restricted).
- **Legacy-derived (`H(k)`) terms are not reduced.**
- The restricted cache is **tied to a rank + `ngroups`**; it is not portable and
  must be used with the matching `ngroups` in `solve`.
- Benefit: steady-state component storage on the `P − ngroups` non-root ranks.

## Decisions (locked with the user)

1. MPI-couple `discretize` (the real slice-on-store).
2. 2b-i first (roots keep full), then 2b-ii (distributed filter).

## Verified background

- `reconstruct`/`reconstruct_all` use only `N_per_var`/`domain`/`derived_var_order`/
  `derived_caches` — never `A_kcomponents`/`B_kcomponents` — so a restricted cache
  is safe for reconstruction.
- 2a's `assemble_rows` already builds each rank's owned rows; the per-rank PETSc
  build + ownership (`MatGetOwnershipRange`) is in place.
- `discretize` lives in `src` (pure, MPI-free); the MPI-coupled variant must live
  in the extension.

## Components

### A. `row_range` field on `DiscretizationCache` (src/discretize.jl)
Add a final field `row_range::Union{Nothing,Tuple{Int,Int}}` (default `nothing` =
full/replicated, today's behavior). The struct's auto all-field constructor gains
the field; the existing outer convenience constructor (computing `A_kcomponents`)
passes `nothing`, so `discretize` is unchanged.

### B. `restrict_cache_rows(cache, rstart, rend)` (src/mpi_prep.jl, pure)
Returns a new cache with `A_kcomponents`/`B_kcomponents` sliced to `[rstart+1:rend,:]`
(now `nrows × N`), `row_range=(Int(rstart),Int(rend))`, all other fields kept
(`derived_caches` full — k-dependent). Errors if `cache.row_range !== nothing`
(already restricted). `A_components`/`B_components` (legacy pre-k dicts, unused by
`assemble`/`assemble_rows`) pass through unchanged.

### C. `assemble_rows` branch (src/mpi_prep.jl)
- `cache.row_range === nothing` → 2a path (slice full components `Ap[(rstart+1):rend,:]`).
- else → assert `(Int(rstart),Int(rend)) == cache.row_range`; components are already
  owned-row slices → **sum directly**. Derived-term handling (`_place_in_block_rows`
  on full derived matrices) is shared/unchanged.

### D. `discretize_distributed(prob; ngroups=1, kwargs...)` (ext/BiGSTARSMPIExt.jl)
```julia
function BiGSTARS.discretize_distributed(prob; ngroups::Integer=1, kwargs...)
    MPI.Initialized() || MPI.Init()
    world = MPI.COMM_WORLD; P = MPI.Comm_size(world); wrank = MPI.Comm_rank(world)
    (1 ≤ ngroups ≤ P) || error("ngroups=$ngroups must be in 1:$P")
    (P % ngroups == 0) || error("nprocs=$P not divisible by ngroups=$ngroups")
    cache = BiGSTARS.discretize(prob; kwargs...)              # full, every rank
    psize = P ÷ ngroups
    grank = wrank % psize                                     # rank within its group (deterministic)
    # Owned rows via PETSc's deterministic PETSC_DECIDE split as a PURE FORMULA —
    # NOT a PETSc probe. A probe would call MatCreate before any PetscInitialize
    # (solve inits SLEPc, not discretize_distributed) → C-level abort; and pre-initing
    # SLEPc here would collide with solve's once-per-process options guard. The
    # formula needs no PETSc at all. solve's per-group build Mat uses the same split;
    # assemble_rows asserts the resulting range == row_range.
    rstart, rend = _petsc_ownership(cache.N_total, psize, grank)
    # Group roots keep the FULL cache (for the mass filter); others restrict + free full.
    return grank == 0 ? cache : restrict_cache_rows(cache, rstart, rend)
end
```
Declared as `function discretize_distributed end` in `src` (a stub that errors with
the install hint when the extension is not loaded, mirroring `solve`), exported.

### E. `solve` — unchanged
`solve(cache; ngroups)` already (2a) builds each rank's matrix via
`assemble_rows(cache, k, rstart, rend)` and filters with `_assemble_B_full(cache,k)`
on the group root. With a `discretize_distributed` cache: non-roots have
`row_range` set → `assemble_rows` direct-sum (asserting solve's `[rstart,rend)` ==
`row_range`, which matches since same `N`+`ngroups`); roots have the full cache →
`assemble_rows` slices, and `_assemble_B_full` gets the full B. **`ngroups` passed
to `solve` must equal the one passed to `discretize_distributed`** (documented;
mismatch trips the `row_range` assert rather than silently corrupting).

## Testing
- **Cross-platform (pure)** in `test/test_mpi_prep.jl`: for plain/augmented/legacy
  caches and several ranges, `assemble_rows(restrict_cache_rows(c,rs,re),k,rs,re) ≈
  assemble(c,k)[rs+1:re,:]` and `≈ assemble_rows(c,k,rs,re)` (2a path); double
  `restrict_cache_rows` throws; `reconstruct` works on a restricted cache.
- **MPI CI** (`mpi.yml`): add to `test/mpi/test_slepc.jl` a run using
  `discretize_distributed(prob; ngroups=ng)` then `solve(cache; ngroups=ng)` and
  assert the same analytic `σ₁=k²+π²` as the full-cache path. Verifies the mixed
  full-root / restricted-non-root caches produce identical results at n=1/2/4.

## Risks
- Ownership match between the `_petsc_ownership` formula (in `discretize_distributed`)
  and `solve`'s build Mats (`PETSC_DECIDE`) relies on the formula replicating PETSc's
  `PetscSplitOwnership` for the same `N` + group size; the `row_range` assert in
  `assemble_rows` catches any mismatch loudly (collective — all non-roots assert the
  same). `_petsc_ownership` is unit-tested cross-platform.
- `discretize_distributed` + `solve` must use the same `ngroups` — documented;
  asserted.
- Peak memory unchanged; roots keep full (extreme-N-on-root → 2b-ii).
- New `row_range` mode in the core cache — only `assemble_rows` consumes the
  restricted shape; `assemble`/`reconstruct` only ever see `nothing` (full).

## Out of scope (→ 2b-ii)
- Distributed mass filter (PETSc `MatMult`/`VecNorm`) so group roots are also
  restricted (true all-ranks-restricted / extreme-N-on-every-node).
