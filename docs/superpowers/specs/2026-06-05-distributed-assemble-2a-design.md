# Design: Phase 2a — row-distributed assemble from replicated components

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Builds on:** v4.0.0 (SLEPc sole eigensolver) + Phase 1 (MPI groups, `ngroups`).

## Goal

Each MPI rank builds **only its owned rows** of the pencil `A, B` — by slicing the
already-replicated `DiscretizationCache` components and computing the owned rows of
the derived-variable terms — and inserts them directly into the distributed PETSc
matrix. This eliminates the **rank-0 full-matrix build** and the **MPI CSR
scatter** that the current `_build_petsc_mat` performs.

It is **Phase 2a** of distributed assembly. **Phase 2b (row-distributed component
construction inside `discretize`, so the components themselves are never fully
materialized) is out of scope here.** In 2a the components stay replicated on every
rank (that is `discretize`'s doing, unchanged); 2a removes the comm + the rank-0
assembled copy and establishes the ownership model + test harness that 2b needs.

## Decisions (locked with the user)

1. 2a first (foundation), then 2b — incremental.
2. Full distributed discretize+assemble is the eventual goal; 2a is its prerequisite.

## Background (verified)

`_assemble(cache, k)` (src/discretize.jl:1896) is exactly:
```
A = Σ_p k^p · A_kcomponents[p]   +   Σ_derived  k-weight · place_in_block(coeff·H(k)·rhs, eq,var)
B = Σ_p k^p · B_kcomponents[p]
```
There is **no separate boundary-condition step** — BC rows are baked into the
components — so a row-slice of the components reproduces the BC rows. The cache is
built by `discretize(prob)`, which runs identically on every rank (no MPI), so
every rank already holds the full component set. `H(k) = _sparse_block_inverse(...)`
exists only for the derived terms (and is a global inverse).

## Architecture

```
each rank r (in group_comm):
  MatCreate(comm) → MatSetSizes → MatGetOwnershipRange → [rstart, rend)   # PETSc picks the split
  A_rows, B_rows = assemble_rows(cache, k, rstart, rend)                  # pure, local, no full matrix
  → _to_csr(A_rows) → _csr_block_nnz_split (prealloc) → _insert_rows!     # existing helpers, local rows
  MatAssemblyBegin/End
  (NO MPI.send/recv scatter; NO rank-0 full A)
```

## Components

### A. `assemble_rows(cache, k, rstart, rend)` — pure core (src/mpi_prep.jl)
Returns `(A_rows, B_rows)`, each an `(rend-rstart) × N_total` `SparseMatrixCSC`,
equal to `assemble(cache, k)[rstart+1:rend, :]` but built **without** materializing
any full N×N matrix:
```julia
nrows = rend - rstart
A_rows = spzeros(ComplexF64, nrows, N)
for (kp, Ap) in cache.A_kcomponents
    A_rows .+= _k_coeff(kp, k_vals) .* Ap[rstart+1:rend, :]     # sparse row-slice of replicated component
end
# derived terms, owned-row portion only:
for (dname, dc) in cache.derived_caches, term in dc.terms
    H_k = (build locally, as in _assemble)
    combined = coeff_mat * H_k * rhs_mat                         # N_per_var × N_per_var
    A_rows .+= w · _place_in_block_rows(combined, eq_idx, var_idx, N_per_var, N_vars, rstart, rend)
end
# B analogously (no derived terms in B)
```
- `_k_coeff` and the k_vals dict are computed exactly as in `assemble`/`_assemble`
  (reuse those helpers; factor the k_vals construction so both call it).
- **Pure Julia, no MPI/PETSc** → unit-tested cross-platform against the full
  `assemble`. This is the primary correctness gate.

### B. `_place_in_block_rows(mat, eq_idx, var_idx, N_per_var, N_vars, rstart, rend)` — helper (src/mpi_prep.jl)
Row-restricted form of `place_in_block` (discretize.jl:1231): emits an
`(rend-rstart) × N_total` sparse matrix whose only nonzeros are the rows of the
`eq_idx` block that intersect `[rstart, rend)`, placed at columns
`[(var_idx-1)·N_per_var+1 : var_idx·N_per_var]`, taking the corresponding local
rows of `mat`. Pure; unit-tested (incl. the no-overlap case → all-zero slice).

### C. Distributed PETSc build (ext/BiGSTARSMPIExt.jl)
Replace the body of `_build_petsc_mat` (currently: rank 0 builds full CSR + scatters
row-blocks via `MPI.send`/`recv`) with a per-rank local build:
```julia
function _build_petsc_mat_local(cache, k, which::Symbol, N, comm)  # which ∈ (:A,:B)
    M = MatCreate(comm); MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N, N); MatSetFromOptions(M)
    rstart, rend = MatGetOwnershipRange(M)
    Arows, Brows = assemble_rows(cache, k, rstart, rend)
    rows = which === :A ? Arows : Brows
    rowptr, colind, vals = _to_csr(rows)                 # CSR of the local (nrows × N) slice
    d_nnz, o_nnz = _csr_block_nnz_split(rowptr, colind, 0, rend-rstart, rstart, rend)
    _prealloc!(M, MPI.Comm_size(comm), d_nnz, o_nnz) || MatSetUp(M)
    _insert_rows!(M, rstart, rowptr, colind, vals)
    MatAssemblyBegin/End; return M
end
```
`_to_csr`, `_csr_block_nnz_split`, `_prealloc!`, `_insert_rows!` are reused
unchanged. `assemble_rows` is called once per matrix build; computing both A and B
slices together (one call) and caching is a minor optimization (build both, use
the requested one) — acceptable to call twice for clarity.

### D. Integration (ext)
- `_solve_one_adaptive(cache, k, N, comm; …)` takes `cache, k` instead of
  `A_csr, B_csr`; builds `A`/`B` via `_build_petsc_mat_local`. The σ-loop and gather
  are unchanged.
- `solve` no longer assembles on the group root or computes `_to_csr`/scatter.
  `N = cache.N_total` is read directly on every rank (the cache is replicated), so
  the `MPI.bcast(N, …)` is removed. It calls
  `_solve_one_adaptive(cache, k_values[i], N, group_comm; …)`.
- **Mass filter:** `_solve_one_adaptive` currently filters with
  `sparse_from_csr(B_csr)` on rank 0. With distributed assembly there is no full
  `B_csr`. Rank 0 builds full `B` locally for the filter via the existing
  `assemble(cache, k)` (B part only); B is the cheap mass matrix and the cache is
  local, so no scatter. (2b will revisit this.)

## Data flow / collective-safety
- `N = cache.N_total` is identical on every rank (replicated cache) → no bcast, no
  divergence.
- `MatCreate`/`MatSetSizes`/`MatGetOwnershipRange` are collective on `comm`; every
  rank calls them, gets its own `[rstart,rend)`. Each rank builds its slice
  independently (pure, local) — no inter-rank communication during assembly.
- Insertion + `MatAssembly` are collective on `comm` (all ranks participate). The
  removed `MPI.send`/`recv` scatter is pure win (less comm, no rank-0 serialization).

## Testing
- **Cross-platform unit tests** (`test/test_mpi_prep.jl`):
  - `assemble_rows(cache,k,rstart,rend) == assemble(cache,k)[rstart+1:rend, :]` for
    several `[rstart,rend)` (full range, a middle slice, an empty slice, a
    single row), on: a plain Fourier×Cheb problem, an **augmented-derived** problem,
    and a **legacy-derived** (`augment_derived=false`) problem.
  - `_place_in_block_rows` matches `place_in_block(...)[rstart+1:rend, :]`, incl. a
    non-overlapping range → all-zero.
- **MPI CI** (`mpi.yml`, n=1/2/4): the existing analytic tests (`σ₁=k²+π²`,
  residual, spurious filter, groups) must still pass — now produced via distributed
  assembly with no scatter. No new MPI test needed; the existing suite is the
  integration gate.

## Risks
- `assemble_rows` must reproduce `assemble` exactly — mitigated by the pure
  cross-platform unit test (strong, local; the main gate).
- `H(k)` is built per-rank (replicated compute) for derived problems — correct, but
  no memory win there; that is 2b's concern.
- Rank-0 full-B build for the mass filter — minor (cheap mass matrix, no comm).
- PETSc insertion of locally-built rows is CI-only verifiable.

## Out of scope (→ Phase 2b)
- Row-distributed construction of the components inside `discretize` (so the
  components are never fully materialized). Legacy-derived (`H(k)` inverse) cannot
  be fully row-distributed and will remain replicated/excluded in 2b.
