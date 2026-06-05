# Phase 2b-ii: distributed mass filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the spurious-mode mass `‖Bχ‖/‖χ‖` distributedly (no full `B`), so `discretize_distributed` can restrict EVERY rank (group roots too).

**Architecture:** Per converged eigenpair, the distributed mass is `MatMult(B, vr)` + per-rank `sum(abs2)` of `VecGetArray` slices reduced with `MPI.Allreduce` (no `VecNorm`). A pure `_keep_by_mass` reproduces `_filter_physical_modes`' keep-rule on those masses; the solver filters by it instead of building a full `B`; `discretize_distributed` then restricts all ranks.

**Tech Stack:** Julia, MPI.jl 0.19, PetscWrap (`MatMult`/`VecDuplicate`/`VecGetArray`), complex PETSc/SLEPc (filter path CI-only; `_keep_by_mass` pure + locally tested).

---

## Verification reality
- `_keep_by_mass` is pure → unit-tested locally + cross-platform CI.
- The distributed mass (`_gather_eigenpairs`), the filter swap, and `discretize_distributed` restrict-all are CI-only (`mpi.yml`). Locally: parse + cross-platform suite.
- Julia: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`, `export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"`, Bash `dangerouslyDisableSandbox=true`.

## File structure
```
src/results.jl          modify: + _keep_by_mass (pure keep-rule on masses)
ext/BiGSTARSMPIExt.jl   modify: _gather_eigenpairs computes distributed masses + returns them;
                                _solve_one_adaptive filters via _keep_by_mass; discretize_distributed
                                restricts ALL ranks; import _keep_by_mass, drop _filter_physical_modes/_assemble_B_full
test/test_mpi_prep.jl   modify: + _keep_by_mass unit tests (cross-platform)
test/mpi/test_slepc.jl  modify: + discretize_distributed singular-B (all-ranks-restricted) run
docs/src/mpi.md         modify: note discretize_distributed now restricts every rank
```

---

## Task 1: `_keep_by_mass` pure helper

**Files:** Modify `src/results.jl`, `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append to the top-level testset in `test/test_mpi_prep.jl`:
```julia
@testset "_keep_by_mass keeps physical, drops near-zero" begin
    @test BiGSTARS._keep_by_mass([1.0, 1e-12, 2.0]) == [1, 3]    # middle is spurious (≈0)
    @test BiGSTARS._keep_by_mass([1.0, 2.0, 3.0]) == [1, 2, 3]   # all physical
    @test BiGSTARS._keep_by_mass([5.0]) == [1]                   # single
    @test BiGSTARS._keep_by_mass(Float64[]) == Int[]            # empty
    @test BiGSTARS._keep_by_mass([0.0, 0.0]) == [1, 2]          # all-zero → keep all (fallback)
    @test BiGSTARS._keep_by_mass([1.0, 0.0, 1.0]) == [1, 3]     # exact-zero dropped
end
```

- [ ] **Step 2: Run, verify fail**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: FAIL — `_keep_by_mass` undefined.

- [ ] **Step 3: Implement in `src/results.jl`** (after `_filter_physical_modes`)
```julia
"""
    _keep_by_mass(masses; rtol=1e-6) -> Vector{Int}

Indices of physical modes from precomputed per-mode masses `‖Bχᵢ‖/‖χᵢ‖`: keep
those above `rtol · maximum(masses)` (drop singular-B infinite modes, mass ≈ 0).
The keep-rule of `_filter_physical_modes`, but on masses computed distributedly.
Keeps everything when the set is empty / all-zero / nothing would survive.
"""
function _keep_by_mass(masses::AbstractVector{<:Real}; rtol::Float64=1e-6)
    isempty(masses) && return Int[]
    scale = maximum(masses)
    scale == 0.0 && return collect(eachindex(masses))
    keep = findall(>(rtol * scale), masses)
    isempty(keep) && return collect(eachindex(masses))
    return keep
end
```

- [ ] **Step 4: Run, verify pass**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/results.jl test/test_mpi_prep.jl
git commit -m "feat: _keep_by_mass (mass-based spurious-mode keep-rule)"
```

---

## Task 2: distributed masses + filter swap + restrict-all (extension)

**Files:** Modify `ext/BiGSTARSMPIExt.jl`. CI-only; locally parse + suite.

- [ ] **Step 1: Update imports**

Replace the `using BiGSTARS: …` block with (add `_keep_by_mass`; drop `_filter_physical_modes`, `_assemble_B_full`):
```julia
using BiGSTARS: _to_csr, _csr_block_nnz_split, _eps_options,
                _sigma_schedule, _group_indices, _petsc_ownership, assemble_rows,
                SolverResults, ConvergenceHistory, _keep_by_mass,
                sort_eigenvalues!, restrict_cache_rows
```

- [ ] **Step 2: Compute distributed masses in `_gather_eigenpairs`**

Replace the whole `_gather_eigenpairs` function with (new signature takes `B`, returns `masses`):
```julia
function _gather_eigenpairs(eps, A, B, nconv::Integer, N::Integer, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)
    rstart, rend = MatGetOwnershipRange(A)
    nlocal = rend - rstart
    counts = Cint.(MPI.Allgather(Int(nlocal), comm))

    λ = rank == 0 ? Vector{ComplexF64}(undef, nconv) : ComplexF64[]
    Χ = rank == 0 ? Matrix{ComplexF64}(undef, N, nconv) : zeros(ComplexF64, 0, 0)
    masses = Vector{Float64}(undef, nconv)         # replicated on every rank (via Allreduce)

    vr, vi = MatCreateVecs(A)
    Bvr = VecDuplicate(vr)
    for ie in 0:(nconv - 1)
        vpr, vpi, _, _ = EPSGetEigenpair(eps, ie, vr, vi)
        local_arr, local_ref = VecGetArray(vr)     # this rank's owned entries
        sendbuf = Vector{ComplexF64}(local_arr)
        nv2 = sum(abs2, local_arr)                 # ‖vr‖² (local part)
        VecRestoreArray(vr, local_ref)             # restore before MatMult uses vr

        MatMult(B, vr, Bvr)                         # distributed B·vr
        bl, bref = VecGetArray(Bvr)
        nb2 = sum(abs2, bl)                         # ‖B vr‖² (local part)
        VecRestoreArray(Bvr, bref)

        nv2g = MPI.Allreduce(nv2, +, comm)         # collective (every rank)
        nb2g = MPI.Allreduce(nb2, +, comm)
        masses[ie + 1] = sqrt(nb2g) / sqrt(max(nv2g, eps()))

        if rank == 0
            recvbuf = Vector{ComplexF64}(undef, N)
            MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
            λ[ie + 1] = ComplexF64(vpr)
            Χ[:, ie + 1] = recvbuf
        else
            MPI.Gatherv!(sendbuf, nothing, 0, comm)
        end
    end
    VecDestroy(Bvr)
    VecDestroy(vr)
    VecDestroy(vi)
    return λ, Χ, masses
end
```
(Per-iteration collective sequence on every rank: `MatMult`, `Allreduce`×2, `Gatherv!` — identical count across ranks, so collective-safe.)

- [ ] **Step 3: Use masses + `_keep_by_mass` in `_solve_one_adaptive`**

Change the gather call:
```julia
        λ, Χ, masses = _gather_eigenpairs(eps, A, B, nconv, N, comm)
```
and replace the filter line:
```julia
                λf, Χf = _filter_physical_modes(λ, Χ, _assemble_B_full(cache, Float64(k)))
                λf, Χf = sort_eigenvalues!(λf, Χf, :nearest; σ=σ)
```
with:
```julia
                keep = _keep_by_mass(masses)
                λf, Χf = sort_eigenvalues!(λ[keep], Χ[:, keep], :nearest; σ=σ)
```

- [ ] **Step 4: `discretize_distributed` restricts ALL ranks**

Change the final return:
```julia
    return grank == 0 ? cache : restrict_cache_rows(cache, rstart, rend)
```
to:
```julia
    return restrict_cache_rows(cache, rstart, rend)   # every rank (mass filter is now distributed)
```
(`grank`/`psize` stay — still used by `_petsc_ownership`.)

- [ ] **Step 5: Parse + cross-platform suite**

Run:
```bash
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("ext parse OK")'
"$JL" --project=. -e 'using Pkg; Pkg.test()'
```
Expected: `ext parse OK`; suite PASS (ext not loaded in test env; `_keep_by_mass` tests cover the new logic locally).

- [ ] **Step 6: Commit**
```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: distributed spurious-mass filter; discretize_distributed restricts all ranks"
```

Verification: **CI (mpi.yml)** n=1/2/4.

---

## Task 3: MPI CI test — distributed filter + all-ranks-restricted

**Files:** Modify `test/mpi/test_slepc.jl`. CI-only.

- [ ] **Step 1: Add a distributed-cache singular-B run**

Append to `test/mpi/test_slepc.jl` (after the existing spurious-filter section; `prob2`/`ng`/`rank` are defined earlier there). It uses the SAME static options (`nev=4, which=:LM, tol=1e-10`):
```julia
# 2b-ii: distributed mass filter on an ALL-RANKS-RESTRICTED cache (singular B).
dcache_sp = discretize_distributed(prob2; ngroups=ng, augment_derived=true)
res_dsp = solve(dcache_sp; sigma_0=-0.1, nev=4, which=:LM, tol=1e-10, n_tries=2, ngroups=ng)
if rank == 0
    ts5 = @testset "distributed filter, all-ranks-restricted (ngroups=$(ng))" begin
        @test res_dsp[1].converged
        @test all(e -> abs(e) < 0.5, res_dsp[1].eigenvalues)              # infinite modes dropped
        @test minimum(abs.(res_dsp[1].eigenvalues .- (-1 / π^2))) < 1e-3  # physical n=1 kept
    end
    ts5.anynonpass && exit(1)
end
```

- [ ] **Step 2: Commit**
```bash
git add test/mpi/test_slepc.jl
git commit -m "test(mpi): distributed mass filter on all-ranks-restricted cache (singular B)"
```

Verification: **CI (mpi.yml)** n=1/2/4 — the existing spurious test now exercises the distributed mass filter, and this run additionally restricts every rank.

---

## Task 4: Docs

**Files:** Modify `docs/src/mpi.md`

- [ ] **Step 1:** In the `discretize_distributed` subsection, replace the note that group roots keep the full cache with:
```markdown
Every rank (including group roots) holds only its owned rows: the singular-B
spurious-mode filter is computed distributedly (`MatMult` + `MPI.Allreduce`), so no
rank needs the full mass matrix.
```

- [ ] **Step 2: Commit**
```bash
git add docs/src/mpi.md
git commit -m "docs: discretize_distributed restricts every rank (distributed filter)"
```

---

## Task 5: Final verification

- [ ] **Step 1: Suite + parse**
```bash
"$JL" --project=. -e 'using Pkg; Pkg.test()'
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("OK")'
```
Expected: PASS; `OK`.

- [ ] **Step 2: Push + drive mpi.yml green** (commits/PR only on user authorization). The singular-B tests at n=1/2/4 are the gate for the distributed filter.

---

## Self-review

**Spec coverage:** `_keep_by_mass` pure → T1; distributed mass in `_gather_eigenpairs` → T2 s2; filter swap (`_keep_by_mass`, drop `_assemble_B_full`/`_filter_physical_modes`) → T2 s1/s3; `discretize_distributed` restrict-all → T2 s4; MPI singular-B/all-restricted test → T3; docs → T4. Out-of-scope (matrix-free peak memory) absent. Covered.

**Placeholder scan:** none — full code per step.

**Type consistency:** `_keep_by_mass(masses::AbstractVector{<:Real}; rtol)` defined T1, called T2 s3; `_gather_eigenpairs(eps, A, B, nconv, N, comm) -> (λ, Χ, masses)` new signature defined T2 s2, called T2 s3 with `B` (in scope from `_build_petsc_mats_local`); `discretize_distributed` return uses `restrict_cache_rows(cache, rstart, rend)` (rstart/rend from `_petsc_ownership`, unchanged). `_filter_physical_modes`/`_assemble_B_full` remain defined in `src` (dropped only from the ext import).
```
