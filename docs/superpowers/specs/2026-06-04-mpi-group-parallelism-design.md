# Design: across-wavenumber parallelism via MPI groups (Phase 1)

**Date:** 2026-06-04
**Status:** Approved (pending spec review)
**Builds on:** the SLEPc/PETSc sole-eigensolver rework (v4.0.0).

## Goal

Make the distributed eigensolver parallel **across wavenumbers as well as within
each solve**. Today `solve(cache, k_values; …)` iterates wavenumbers
*sequentially*, all ranks collaborating on one pencil at a time. Phase 1 splits
the ranks into independent **groups**; each group owns a subset of the
wavenumbers and runs the existing distributed solve on its own sub-communicator.

This is Phase 1 of a two-phase "true parallel" effort. **Phase 2 (distributed
assembly — each rank builds only its owned rows) is out of scope here**; within
each group, assembly stays on the group's rank 0.

## Decisions (locked with the user)

1. **Explicit `ngroups` kwarg**, default `1` (no auto-pick). `ngroups=1` reproduces
   today's behavior exactly.
2. **Round-robin** wavenumber distribution across groups.
3. Phase 1 only (groups); distributed assembly deferred to Phase 2.

## Feasibility (verified against installed sources)

PetscWrap 0.1.x and SlepcWrap 0.1.x accept a communicator on object creation, so
sub-communicator solves need **no wrapper changes**:
- `MatCreate(comm::MPI.Comm = MPI.COMM_WORLD)` — `PetscWrap/src/mat.jl:49`
- `EPSCreate(comm::MPI.Comm = MPI.COMM_WORLD)` — `SlepcWrap/src/eps.jl:18`
- `VecCreate(comm)`, `KSPCreate(comm)`, `MatCreateVecs(mat)` (uses `mat.comm`).

The current extension uses the `COMM_WORLD` defaults; Phase 1 threads a group
communicator through instead.

## Architecture

```
COMM_WORLD  (P ranks)
   │  MPI.Comm_split(color = rank ÷ (P÷G))
   ├── group 0  (P÷G ranks)  ── k_values[0], k_values[G],  …   (round-robin)
   ├── group 1  (P÷G ranks)  ── k_values[1], k_values[G+1], …
   └── …  (G groups)
        each group: existing distributed solve on group_comm, sequential over its k's
        results live on the group's root (global rank = group_id·(P÷G))
   │  point-to-point: each group-root → global rank 0
   ▼
Vector{SolverResults}  (length = length(k_values), fully populated on global rank 0)
```

## Components

### A. Public entrypoint — `BiGSTARS.solve` (ext/BiGSTARSMPIExt.jl)
Add `ngroups::Integer = 1`. New top-level flow:

1. `MPI.Initialized() || MPI.Init()`; SLEPc init as today (once per process, all
   ranks, same options string — unchanged).
2. `P = MPI.Comm_size(COMM_WORLD)`. **Validate:** `ngroups ≥ 1`, `ngroups ≤ P`,
   `P % ngroups == 0` — else `error(...)` with a clear message (collective: every
   rank validates the same inputs, so all error together — no deadlock).
3. `psize = P ÷ ngroups`; `color = rank ÷ psize`; `group_comm = MPI.Comm_split(COMM_WORLD, color, rank)`; `group_id = color`.
4. Assign indices: group `g` owns `[i for i in eachindex(k_values) if (i-1) % ngroups == g]` (round-robin, 1-based corrected).
5. For each owned index `i`: assemble on `group_comm`'s rank 0, distributed solve
   on `group_comm` (existing `_solve_one_adaptive`, now comm-parameterized).
   Store into a local `Dict{Int,SolverResults}` on the group root.
6. Gather to global rank 0 (component C). Return `Vector{SolverResults}` of length
   `length(k_values)` — populated on global rank 0; empty markers elsewhere.
7. `ngroups == 1` short-circuits to the current single-`COMM_WORLD` path (group_comm
   == COMM_WORLD, group owns all k's) — behavior identical to v4.0.0.

### B. Comm-parameterize the existing solve path
Thread `comm` (the group comm) through every collective. **Required fix:**
`_build_petsc_mat` currently calls `MatCreate()` (defaults to `COMM_WORLD`) and
`_solve_one_adaptive` calls `EPSCreate()` (defaults to `COMM_WORLD`). Change to
`MatCreate(comm)` / `EPSCreate(comm)`, and ensure `_build_petsc_mat`,
`_gather_eigenpairs`, `_solve_one_adaptive` use the passed `comm` for **all** of
`MPI.Gather`/`Allgather`/`send`/`recv`/`bcast`/`Gatherv!`. `MatCreateVecs(A)` and
`MatGetOwnershipRange(A)` already follow `A`'s comm. At `ngroups=1`,
`comm == COMM_WORLD`, so this is a no-op behavior change for the existing tests.

### C. Result gather (group-roots → global rank 0)
- A rank is a **group root** iff `MPI.Comm_rank(group_comm) == 0` (global rank
  `group_id·psize`).
- Global rank 0 is group 0's root. It first places **its own** group's results
  directly into the output vector, then receives the rest: for `g in 1:(ngroups-1)`,
  `MPI.recv(g·psize, TAG, COMM_WORLD)` → a `Vector{Tuple{Int,SolverResults}}`;
  place each into the output vector.
- Each non-zero group root `MPI.send(its_pairs, 0, TAG, COMM_WORLD)`.
- `SolverResults` serializes over MPI (same mechanism the CSR scatter uses).
- Collective-safety: exactly `ngroups-1` sends matched by `ngroups-1` recvs on
  global rank 0; all other ranks do nothing in the gather. Deterministic counts.

## Collective-safety rules (the one hard discipline)
- Inside a group, **every** MPI/PETSc collective uses `group_comm` — never
  `COMM_WORLD`. A stray `COMM_WORLD` collective inside the per-group solve would
  block waiting for ranks in other groups → deadlock.
- The cross-group result gather is the *only* `COMM_WORLD` communication after the
  split, and it is point-to-point with matched send/recv counts.
- Input validation (step 2) runs identically on all ranks before any split, so a
  bad `ngroups` errors collectively.
- Empty group (more groups than wavenumbers): its ranks own no indices, skip the
  solve loop entirely, and its root sends an empty pair list. No group-internal
  collectives occur, so no divergence.

## Testing

- **Cross-platform (`runtests.jl`)**: `ngroups` is solver-only; no new pure-Julia
  unit beyond an argument-validation note. The round-robin index split is a pure
  function — extract `_group_indices(nk, ngroups, group_id)` into `mpi_prep.jl`
  and unit-test it (cross-platform): partition correctness, coverage (every index
  assigned exactly once), empty-group case.
- **MPI CI (`mpi.yml`)**: add an `mpiexec -n 4` run of `test/mpi/test_slepc.jl`
  with `ngroups=2` (2 groups × 2 ranks) over several wavenumbers (e.g. k ∈
  {0.5, 1.0, 1.5, 2.0}); assert each `k`'s smallest eigenvalue matches analytic
  `σ₁(k) = k² + π²` to tolerance. Keep the existing n=1 / n=2 (`ngroups=1`) runs
  as the no-split regression. The script must guard collectives correctly under
  groups (results only on global rank 0).

## Risks
- **Not locally testable** (no complex PETSc/MPI here) — CI/cluster-only,
  iterative. Multi-communicator code is deadlock-prone; the §"Collective-safety"
  discipline is the mitigation, and the within-group solve reuses already-verified
  v4.0.0 logic.
- `mpiexec -n 4` increases CI time/oversubscription on the runner; use
  `--oversubscribe` (already used for n=2).
- Load imbalance if `length(k_values)` not divisible by `ngroups` — acceptable
  (round-robin minimizes it; some groups do one extra k).

## Out of scope
- **Phase 2: distributed assembly** (each rank builds owned rows) — the rank-0
  per-group assembly/memory bottleneck remains.
- Auto-selecting `ngroups`.
- Nested/uneven group sizes (`P % ngroups ≠ 0` is an error, not handled).
