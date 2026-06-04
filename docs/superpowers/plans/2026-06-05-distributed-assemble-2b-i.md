# Phase 2b-i: MPI-coupled discretize_distributed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `discretize_distributed(prob; ngroups)` returning a per-rank row-restricted `DiscretizationCache` (group roots keep full), so steady-state per-rank component memory drops on non-root ranks.

**Architecture:** A new `row_range` field marks a cache as restricted. Pure `restrict_cache_rows` slices the k-independent components to a rank's owned rows; `assemble_rows` gains a branch (restricted → sum directly; full → slice, as in 2a). The extension's `discretize_distributed` builds the full cache, probes PETSc for each rank's `[rstart,rend)` on its `group_comm`, and returns the restricted (non-root) or full (root) cache. `solve` is unchanged.

**Tech Stack:** Julia, SparseArrays, MPI.jl 0.19, PetscWrap/SlepcWrap 0.1.x, complex PETSc/SLEPc (the `discretize_distributed`/solve path is CI-only; the `restrict_cache_rows`/`assemble_rows` core is pure + locally tested).

---

## Verification reality
- `row_range`, `restrict_cache_rows`, the `assemble_rows` branch are pure → unit-tested locally + cross-platform CI.
- `discretize_distributed` (ext) + solve integration are CI-only (`mpi.yml`). Locally: parse + cross-platform suite.
- Julia: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`, `export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"`, Bash `dangerouslyDisableSandbox=true`.

## File structure
```
src/discretize.jl       modify: + row_range field on DiscretizationCache; outer ctor passes nothing
src/mpi_prep.jl         modify: + restrict_cache_rows; assemble_rows gains the row_range branch
src/solve.jl            modify: + discretize_distributed generic + install-hint fallback
src/BiGSTARS.jl         modify: export discretize_distributed
ext/BiGSTARSMPIExt.jl   modify: + BiGSTARS.discretize_distributed (full→restrict per rank)
test/test_mpi_prep.jl   modify: + restrict_cache_rows / assemble_rows-restricted / reconstruct tests
test/mpi/test_slepc.jl  modify: + discretize_distributed path (CI)
docs/src/mpi.md         modify: document discretize_distributed
```

---

## Task 1: `row_range` field on `DiscretizationCache`

**Files:** Modify `src/discretize.jl`

- [ ] **Step 1: Add the field**

In the `struct DiscretizationCache` block (src/discretize.jl), add a final field after `derived_var_order::Vector{Symbol}`:
```julia
    derived_var_order::Vector{Symbol}
    row_range::Union{Nothing, Tuple{Int, Int}}
```

- [ ] **Step 2: Pass `nothing` from the outer constructor**

In the outer convenience constructor, change its final return:
```julia
    return DiscretizationCache(A_components, B_components, A_kcomponents, B_kcomponents,
                               derived_caches, N_total, N_per_var, N_vars, domain,
                               derived_var_order, nothing)
```

- [ ] **Step 3: Fix any other direct all-field constructor callers**

Run:
```bash
grep -rnE "DiscretizationCache\(" src test | grep -v "::DiscretizationCache"
```
The only all-field (positional) caller should be the outer constructor (just fixed). If any other call passes the full positional field list, append `, nothing`. (The 8-arg outer constructor calls are unaffected.)

- [ ] **Step 4: Verify no behavior change**

Run: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia; export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"; "$JL" --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS, same count (~910). Field defaults to `nothing` everywhere.

- [ ] **Step 5: Commit**
```bash
git add src/discretize.jl
git commit -m "feat: add row_range field to DiscretizationCache (nothing = full)"
```

---

## Task 2: `restrict_cache_rows` + `assemble_rows` branch

**Files:** Modify `src/mpi_prep.jl`, `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append to the top-level testset in `test/test_mpi_prep.jl`:
```julia
@testset "restrict_cache_rows + restricted assemble_rows" begin
    function mk_aug()
        dom = Domain(z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:psi], eigenvalue=:sigma)
        @derive p v dz(dz(v)) = psi
        @derive_bc p v left(v) == 0
        @derive_bc p v right(v) == 0
        @equation p sigma * psi == v
        @bc p left(psi) == 0
        @bc p right(psi) == 0
        discretize(p; augment_derived=true)
    end
    function mk_plain()
        dom = Domain(x=FourierTransformed(), z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:u], eigenvalue=:sigma)
        @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc p left(u) == 0
        @bc p right(u) == 0
        discretize(p)
    end
    for (mk, k) in ((mk_plain, 1.3), (mk_aug, 0.0))
        cache = mk(); N = cache.N_total
        @test cache.row_range === nothing
        for (rs, re) in ((0, N), (0, cld(N,2)), (cld(N,2), N), (N-1, N))
            rc = BiGSTARS.restrict_cache_rows(cache, rs, re)
            @test rc.row_range == (rs, re)
            Ar, Br = BiGSTARS.assemble_rows(rc, k, rs, re)                  # restricted (direct sum)
            Af, Bf = BiGSTARS.assemble_rows(cache, k, rs, re)              # full (2a slice)
            @test Ar ≈ Af && Br ≈ Bf
            @test_throws ArgumentError BiGSTARS.assemble_rows(rc, k, 0, re ÷ 2)  # range mismatch
        end
        # double restriction is rejected
        rc = BiGSTARS.restrict_cache_rows(cache, 0, N)
        @test_throws ArgumentError BiGSTARS.restrict_cache_rows(rc, 0, N)
    end
end
```

- [ ] **Step 2: Run, verify fail**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: FAIL — `restrict_cache_rows` undefined / `row_range` branch absent.

- [ ] **Step 3: Implement `restrict_cache_rows` in `src/mpi_prep.jl`**

```julia
"""
    restrict_cache_rows(cache, rstart, rend) -> DiscretizationCache

Return a cache whose k-independent components (`A_kcomponents`/`B_kcomponents`) are
sliced to the owned global rows `[rstart,rend)` (0-based half-open), with
`row_range=(rstart,rend)` set. `derived_caches` are kept full (k-dependent `H(k)`).
Errors if `cache` is already restricted. Pure (no MPI/PETSc).
"""
function restrict_cache_rows(cache, rstart::Integer, rend::Integer)
    cache.row_range === nothing ||
        throw(ArgumentError("cache already restricted to $(cache.row_range)"))
    function _slice_rows(d)
        out = empty(d)
        for (kp, M) in d
            out[kp] = M[(rstart + 1):rend, :]
        end
        return out
    end
    return DiscretizationCache(cache.A_components, cache.B_components,
        _slice_rows(cache.A_kcomponents), _slice_rows(cache.B_kcomponents),
        cache.derived_caches, cache.N_total, cache.N_per_var, cache.N_vars,
        cache.domain, cache.derived_var_order, (Int(rstart), Int(rend)))
end
```

- [ ] **Step 4: Add the `row_range` branch to `assemble_rows`**

Replace the existing `assemble_rows` head + the two component loops. The new version
validates the range against `row_range` and slices only when the cache is full:
```julia
function assemble_rows(cache, k::Float64,
                       rstart::Integer, rend::Integer)
    N = cache.N_total
    (0 ≤ rstart ≤ rend ≤ N) ||
        throw(ArgumentError("row range [$rstart,$rend) out of [0,$N]"))
    if cache.row_range !== nothing
        cache.row_range == (Int(rstart), Int(rend)) ||
            throw(ArgumentError("requested rows ($rstart,$rend) ≠ cache.row_range $(cache.row_range)"))
    end
    sliced = cache.row_range === nothing          # full cache → slice; restricted → use as-is
    k_vals = _k_values(cache, k)
    nrows = rend - rstart

    A_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Ap) in cache.A_kcomponents
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        A_rows = A_rows + c * (sliced ? Ap[(rstart + 1):rend, :] : Ap)
    end

    B_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Bp) in cache.B_kcomponents
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        B_rows = B_rows + c * (sliced ? Bp[(rstart + 1):rend, :] : Bp)
    end

    for (_, dc) in cache.derived_caches
        isempty(dc.terms) && continue
        op_k = dc.op_k0
        for (kp, mat) in dc.op_k_components
            c = _k_coeff(kp, k_vals); c == 0.0 && continue
            op_k = op_k + c * mat
        end
        H_k = _sparse_block_inverse(op_k, cache.domain; bcs=dc.bcs)
        for (eq_idx, var_idx, total_kp, coeff_mat, rhs_mat) in dc.terms
            w = _k_coeff(total_kp, k_vals); w == 0.0 && continue
            combined = coeff_mat * H_k * rhs_mat
            A_rows = A_rows + w * _place_in_block_rows(combined, eq_idx, var_idx,
                                                       cache.N_per_var, cache.N_vars,
                                                       rstart, rend)
        end
    end
    return A_rows, B_rows
end
```
(The derived block is unchanged — `derived_caches` are full in both modes, and
`_place_in_block_rows` already produces only the owned rows.)

- [ ] **Step 5: Run, verify pass**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: restrict_cache_rows + row_range-aware assemble_rows"
```

---

## Task 3: `discretize_distributed` (stub + export + extension)

**Files:** Modify `src/solve.jl`, `src/BiGSTARS.jl`, `ext/BiGSTARSMPIExt.jl`. Ext is CI-only; locally parse + suite.

- [ ] **Step 1: Add the generic + fallback in `src/solve.jl`**

Append to `src/solve.jl`:
```julia
"""
    discretize_distributed(prob; ngroups=1, kwargs...) -> DiscretizationCache

MPI-coupled discretize: builds the full cache, then returns a per-rank row-restricted
cache (group roots keep the full cache, for the singular-B mass filter). Provided by
the extension `BiGSTARSMPIExt` (requires MPI, PetscWrap, SlepcWrap + a complex-scalar
PETSc/SLEPc). Use the SAME `ngroups` in the subsequent `solve`. `kwargs` are forwarded
to `discretize` (e.g. `augment_derived`). Run under `mpiexec -n P julia …`.
"""
function discretize_distributed end

function discretize_distributed(@nospecialize(args...); kwargs...)
    error("BiGSTARS.discretize_distributed requires the SLEPc/PETSc backend: install " *
          "and import MPI, PetscWrap, and SlepcWrap, plus a complex-scalar PETSc/SLEPc " *
          "build. See docs/src/mpi.md.")
end
```

- [ ] **Step 2: Export it (`src/BiGSTARS.jl`)**

Add `discretize_distributed` to the export list (next to `solve`):
```julia
        solve,
        discretize_distributed,
```

- [ ] **Step 3: Implement in the extension (`ext/BiGSTARSMPIExt.jl`)**

Add (import `restrict_cache_rows` in the `using BiGSTARS:` list first), then:
```julia
function BiGSTARS.discretize_distributed(prob; ngroups::Integer=1, kwargs...)
    MPI.Initialized() || MPI.Init()
    world = MPI.COMM_WORLD
    P = MPI.Comm_size(world); wrank = MPI.Comm_rank(world)
    (1 ≤ ngroups ≤ P) || error("discretize_distributed: ngroups=$(ngroups) must be in 1:$(P)")
    (P % ngroups == 0) || error("discretize_distributed: nprocs=$(P) not divisible by ngroups=$(ngroups)")

    cache = BiGSTARS.discretize(prob; kwargs...)          # full, on every rank
    psize = P ÷ ngroups
    group_id = wrank ÷ psize
    group_comm = ngroups == 1 ? world : MPI.Comm_split(world, group_id, wrank)
    grank = MPI.Comm_rank(group_comm)

    # Collective probe (ALL ranks) for this rank's owned rows on the group pencil.
    probe = MatCreate(group_comm)
    MatSetSizes(probe, PETSC_DECIDE, PETSC_DECIDE,
                PetscInt(cache.N_total), PetscInt(cache.N_total))
    MatSetFromOptions(probe)
    rstart, rend = MatGetOwnershipRange(probe)
    MatDestroy(probe)

    # Group roots keep the FULL cache (the mass filter needs full B); others restrict.
    return grank == 0 ? cache : restrict_cache_rows(cache, rstart, rend)
end
```
Add `restrict_cache_rows` to the extension's `using BiGSTARS: …` import list.

- [ ] **Step 4: Parse + cross-platform suite**

Run:
```bash
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("ext parse OK")'
"$JL" --project=. -e 'using Pkg; Pkg.test()'
```
Expected: `ext parse OK`; suite PASS. Also verify the no-backend fallback:
`"$JL" --project=. -e 'using BiGSTARS; try; discretize_distributed(nothing); catch e; println(occursin("SlepcWrap", sprint(showerror,e))); end'` → prints `true`. (Add this as a cross-platform test in `test/test_mpi_prep.jl` if not already covered.)

- [ ] **Step 5: Commit**
```bash
git add src/solve.jl src/BiGSTARS.jl ext/BiGSTARSMPIExt.jl
git commit -m "feat: discretize_distributed — per-rank row-restricted cache (roots full)"
```

---

## Task 4: MPI CI test for `discretize_distributed`

**Files:** Modify `test/mpi/test_slepc.jl`. CI-only.

- [ ] **Step 1: Add a distributed-cache section**

Append to `test/mpi/test_slepc.jl` (reuses the Poisson problem; SAME static options
as the other solves — once-per-process):
```julia
# ── discretize_distributed (Phase 2b-i): per-rank row-restricted cache ─────────
# Build a per-rank restricted cache and solve from it; results must match the
# full-cache analytic reference σ_1(k) = k² + π².
nprocs2 = MPI.Comm_size(MPI.COMM_WORLD)
ngd = (nprocs2 % 2 == 0) ? 2 : 1
dcache = discretize_distributed(prob; ngroups=ngd)
res_d = solve(dcache, [1.0]; sigma_0=10.0, nev=4, which=:LM, tol=1e-10, ngroups=ngd)
if rank == 0
    ts4 = @testset "discretize_distributed (ngroups=$(ngd))" begin
        @test res_d[1].converged
        @test isapprox(minimum(real, res_d[1].eigenvalues), 1.0 + π^2; rtol=1e-4)
    end
    ts4.anynonpass && exit(1)
end
```
(`prob` is the Poisson EVP already built at the top of the file.)

- [ ] **Step 2: Commit**
```bash
git add test/mpi/test_slepc.jl
git commit -m "test(mpi): discretize_distributed per-rank restricted cache at n=1/2/4"
```

Verification: **CI (mpi.yml)** n=1/2/4 — the restricted-cache solve must match the
analytic reference, confirming the mixed full-root / restricted-non-root caches
produce identical results.

---

## Task 5: Document `discretize_distributed`

**Files:** Modify `docs/src/mpi.md`

- [ ] **Step 1: Add a subsection** after the `ngroups` section:
```markdown
### Lower per-rank memory: `discretize_distributed`

For very large problems, build the cache with `discretize_distributed(prob; ngroups=G)`
instead of `discretize(prob)`. It returns, on each non-root rank, a cache holding only
that rank's owned rows of the operator components (group roots keep the full cache for
the spurious-mode filter), reducing steady-state per-rank memory. Pass the **same**
`ngroups` to `solve`:

```julia
cache = discretize_distributed(prob; ngroups=4)
results = solve(cache, k_values; sigma_0=0.02, nev=5, ngroups=4)
```

Notes: the full cache is still built transiently on every rank (peak memory is
unchanged), legacy `augment_derived=false` terms are not reduced, and the returned
cache is tied to this rank + `ngroups` (not portable). A plain `discretize` cache
works with `solve` exactly as before.
```

- [ ] **Step 2: Commit**
```bash
git add docs/src/mpi.md
git commit -m "docs: document discretize_distributed"
```

---

## Task 6: Final verification

- [ ] **Step 1: Cross-platform suite + ext parse**
```bash
"$JL" --project=. -e 'using Pkg; Pkg.test()'
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("OK")'
```
Expected: PASS; `OK`.

- [ ] **Step 2: Push + drive mpi.yml green** (commits/PR only on user authorization — auto-sync + no-commit rule). n=1/2/4 with `discretize_distributed` is the integration gate.

---

## Self-review

**Spec coverage:** `row_range` field → T1; `restrict_cache_rows` → T2; `assemble_rows` branch → T2; `discretize_distributed` (stub+export+ext, roots-keep-full, collective probe) → T3; reconstruct-safe (verified, no code needed) → covered by design; MPI CI → T4; docs → T5. Out-of-scope (2b-ii distributed filter) absent. Covered.

**Placeholder scan:** none — full code per step, exact paths/commands.

**Type consistency:** `row_range::Union{Nothing,Tuple{Int,Int}}` defined T1, set by `restrict_cache_rows` (11-arg constructor, field order matches struct) T2, read by `assemble_rows` branch T2 and `discretize_distributed` T3; `restrict_cache_rows(cache, rstart, rend)` defined T2, called in T3; `discretize_distributed(prob; ngroups, kwargs...)` generic (T3 s1) / ext method (T3 s3) / call (T4) consistent; `assemble_rows(cache, k::Float64, rstart, rend)` signature unchanged (branch internal). `solve` untouched.
