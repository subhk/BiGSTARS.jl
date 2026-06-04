# Phase 2a: row-distributed assemble — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each MPI rank builds only its owned rows of `A,B` by slicing the replicated cache components (+ owned rows of derived terms) and inserts them into the distributed PETSc matrix — removing the rank-0 full-matrix build and the MPI CSR scatter.

**Architecture:** A pure `assemble_rows(cache, k, rstart, rend)` reproduces `assemble(cache,k)[rstart+1:rend,:]` without materializing any full matrix (it slices each `A_kcomponents`/`B_kcomponents` to the owned rows and adds the owned-row portion of each derived term). The extension creates the PETSc `MatMPIAIJ`, reads each rank's `[rstart,rend)`, builds those rows locally, and inserts — no `MPI.send`/`recv` scatter.

**Tech Stack:** Julia, SparseArrays, MPI.jl 0.19, PetscWrap/SlepcWrap 0.1.x, complex PETSc/SLEPc (insertion path CI-only; the `assemble_rows` core is pure and locally unit-tested).

---

## Verification reality
- `_k_values`, `_place_in_block_rows`, `assemble_rows`, `_assemble_B_full` are pure Julia → unit-tested locally + cross-platform CI (the primary correctness gate).
- The PETSc build/insert path (`_build_petsc_mats_local`) is CI-only (`mpi.yml`). Locally: parse-check + cross-platform suite.
- Julia invocation: `JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia`, `export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"`, Bash with `dangerouslyDisableSandbox=true`.

## File structure
```
src/discretize.jl       modify: extract _k_values; assemble(cache,k) calls it (DRY, no behavior change)
src/mpi_prep.jl         modify: + _place_in_block_rows, assemble_rows, _assemble_B_full (pure)
ext/BiGSTARSMPIExt.jl   modify: + _fill_mat!/_build_petsc_mats_local; _solve_one_adaptive(cache,k,…);
                                 solve drops rank-0 assemble + scatter + N-bcast; filter via _assemble_B_full;
                                 import the new helpers. Remove the old _build_petsc_mat (scatter) path.
test/test_mpi_prep.jl   modify: + unit tests for _place_in_block_rows / assemble_rows / _assemble_B_full
```

---

## Task 1: Extract `_k_values` (DRY) in `src/discretize.jl`

**Files:** Modify `src/discretize.jl`

- [ ] **Step 1: Add `_k_values` and route `assemble` through it**

Replace the existing `assemble(cache::DiscretizationCache, k::Float64)` (discretize.jl:1884-1894) with:

```julia
"""
    _k_values(cache, k) -> Dict{Symbol,Float64}

Build the wavenumber dict for a single scalar `k`: `:_total_k => k` when there are
no FourierTransformed directions, else `Symbol(:k_, dim) => k` for each. Shared by
`assemble` and the row-wise `assemble_rows` so they agree exactly.
"""
function _k_values(cache::DiscretizationCache, k::Float64)
    k_vals = Dict{Symbol, Float64}()
    if isempty(cache.domain.transformed_dims)
        k_vals[:_total_k] = k
    else
        for dim in cache.domain.transformed_dims
            k_vals[Symbol(:k_, dim)] = k
        end
    end
    return k_vals
end

function assemble(cache::DiscretizationCache, k::Float64)
    return _assemble(cache, _k_values(cache, k))
end
```

- [ ] **Step 2: Verify no behavior change (cross-platform suite)**

Run:
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS, same count as before (pure refactor).

- [ ] **Step 3: Commit**
```bash
git add src/discretize.jl
git commit -m "refactor: extract _k_values shared by assemble and assemble_rows"
```

---

## Task 2: `_place_in_block_rows` helper

**Files:** Modify `src/mpi_prep.jl`, `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append inside the top-level `@testset` in `test/test_mpi_prep.jl`:

```julia
@testset "_place_in_block_rows matches place_in_block row-slice" begin
    Npv, Nvars = 4, 3
    N = Npv * Nvars
    mat = sparse(ComplexF64[ (10i + j) for i in 1:Npv, j in 1:Npv ])  # dense-ish block
    full = BiGSTARS.place_in_block(mat, 2, 3, Nvars, Npv)             # eq 2, var 3
    for (rs, re) in ((0, N), (Npv, 2Npv), (0, Npv), (5, 7), (2Npv, N))
        got = BiGSTARS._place_in_block_rows(mat, 2, 3, Npv, Nvars, rs, re)
        @test got == full[(rs+1):re, :]
    end
    # block that does not overlap the owned rows → all-zero slice
    z = BiGSTARS._place_in_block_rows(mat, 1, 1, Npv, Nvars, 2Npv, N)
    @test nnz(z) == 0 && size(z) == (N - 2Npv, N)
end
```

- [ ] **Step 2: Run, verify fail**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: FAIL — `_place_in_block_rows` undefined.

- [ ] **Step 3: Implement in `src/mpi_prep.jl`**

Add (near the other helpers):

```julia
"""
    _place_in_block_rows(mat, eq_idx, var_idx, N_per_var, N_vars, rstart, rend) -> SparseMatrixCSC

Row-restricted `place_in_block`: an `(rend-rstart) × (N_per_var*N_vars)` sparse
matrix equal to `place_in_block(mat, eq_idx, var_idx, N_vars, N_per_var)[rstart+1:rend, :]`.
`rstart`/`rend` are 0-based half-open global rows. Only the `eq_idx` block rows that
intersect `[rstart,rend)` carry `mat`'s rows, at the `var_idx` column block.
"""
function _place_in_block_rows(mat::AbstractMatrix, eq_idx::Integer, var_idx::Integer,
                              N_per_var::Integer, N_vars::Integer,
                              rstart::Integer, rend::Integer)
    N_total = N_per_var * N_vars
    nrows = rend - rstart
    bs = (eq_idx - 1) * N_per_var          # 0-based block row start
    be = eq_idx * N_per_var
    cs = (var_idx - 1) * N_per_var         # 0-based block col start
    lo = max(bs, rstart); hi = min(be, rend)
    I = Int[]; J = Int[]; V = ComplexF64[]
    if lo < hi
        sub = mat[(lo - bs + 1):(hi - bs), :]      # mat rows for the overlap (1-based)
        rv = rowvals(sub); nz = nonzeros(sub)
        for col in 1:size(sub, 2)
            for idx in nzrange(sub, col)
                push!(I, rv[idx] + (lo - rstart))  # 1-based row in the output slice
                push!(J, cs + col)                 # global column (1-based)
                push!(V, nz[idx])
            end
        end
    end
    return sparse(I, J, V, nrows, N_total)
end
```

- [ ] **Step 4: Run, verify pass**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: _place_in_block_rows (row-restricted block placement)"
```

---

## Task 3: `assemble_rows` + `_assemble_B_full`

**Files:** Modify `src/mpi_prep.jl`, `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append to `test/test_mpi_prep.jl`. These build real caches via the DSL and compare
to the full `assemble`:

```julia
@testset "assemble_rows == assemble row-slice" begin
    function mkcache_plain()
        dom = Domain(x=FourierTransformed(), z=Chebyshev(N=12, lower=0.0, upper=1.0))
        p = EVP(dom, variables=[:u], eigenvalue=:sigma)
        @equation p sigma * u == -dx(dx(u)) - dz(dz(u))
        @bc p left(u) == 0
        @bc p right(u) == 0
        discretize(p)
    end
    function mkcache_augmented()
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
    function mkcache_legacy()
        dom = Domain(x=FourierTransformed(), y=Fourier(8,[0.0,1.0]), z=Chebyshev(8,[0.0,1.0]))
        p = EVP(dom, variables=[:w,:zeta], eigenvalue=:sigma)
        @derive p v dx(dx(v)) + dy(dy(v)) = dy(dz(w)) - dx(zeta)
        @equation p sigma * w == v - dz(dz(w))
        @equation p sigma * zeta == dz(w)
        @bc p left(w) == 0; @bc p right(w) == 0
        @bc p left(dz(zeta)) == 0; @bc p right(dz(zeta)) == 0
        discretize(p; augment_derived=false)
    end
    for (mk, k) in ((mkcache_plain, 1.3), (mkcache_augmented, 0.0), (mkcache_legacy, 1.0))
        cache = mk()
        Afull, Bfull = assemble(cache, k)
        N = cache.N_total
        for (rs, re) in ((0, N), (0, 0), (0, cld(N,3)), (cld(N,3), 2*cld(N,3)), (N-1, N))
            A_rows, B_rows = BiGSTARS.assemble_rows(cache, k, rs, re)
            @test size(A_rows) == (re - rs, N) && size(B_rows) == (re - rs, N)
            @test A_rows ≈ Afull[(rs+1):re, :]
            @test B_rows ≈ Bfull[(rs+1):re, :]
        end
        @test BiGSTARS._assemble_B_full(cache, k) ≈ Bfull
    end
end
```

- [ ] **Step 2: Run, verify fail**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: FAIL — `assemble_rows`/`_assemble_B_full` undefined.

- [ ] **Step 3: Implement in `src/mpi_prep.jl`**

```julia
"""
    assemble_rows(cache, k, rstart, rend) -> (A_rows, B_rows)

Owned-row slice of the assembled pencil: each is an `(rend-rstart) × N_total`
`SparseMatrixCSC` equal to `assemble(cache,k)[rstart+1:rend, :]`, built WITHOUT
materializing any full N×N matrix. `rstart`/`rend` are 0-based half-open global rows.
Mirrors `_assemble` (discretize.jl) term-by-term, sliced to the owned rows.
"""
function assemble_rows(cache::DiscretizationCache, k::Float64,
                       rstart::Integer, rend::Integer)
    N = cache.N_total
    (0 ≤ rstart ≤ rend ≤ N) ||
        throw(ArgumentError("row range [$rstart,$rend) out of [0,$N]"))
    k_vals = _k_values(cache, k)
    nrows = rend - rstart

    A_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Ap) in cache.A_kcomponents
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        A_rows = A_rows + c * Ap[(rstart + 1):rend, :]
    end

    B_rows = spzeros(ComplexF64, nrows, N)
    for (kp, Bp) in cache.B_kcomponents
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        B_rows = B_rows + c * Bp[(rstart + 1):rend, :]
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

"""
    _assemble_B_full(cache, k) -> SparseMatrixCSC

The full mass matrix `B = Σ_p k^p · B_kcomponents[p]` (B has no derived terms).
Cheap; built on the group root for the singular-`B` spurious-mode filter when the
distributed path means no rank holds a full `B`.
"""
function _assemble_B_full(cache::DiscretizationCache, k::Float64)
    k_vals = _k_values(cache, k)
    N = cache.N_total
    B = spzeros(ComplexF64, N, N)
    for (kp, Bp) in cache.B_kcomponents
        c = _k_coeff(kp, k_vals); c == 0.0 && continue
        B = B + c * Bp
    end
    return B
end
```

- [ ] **Step 4: Run, verify pass**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: PASS (plain, augmented, and legacy caches; all ranges; B-full).

- [ ] **Step 5: Commit**
```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: assemble_rows + _assemble_B_full (row-distributed assembly core)"
```

---

## Task 4: Distributed build in the extension (no scatter)

**Files:** Modify `ext/BiGSTARSMPIExt.jl`. CI-only verification; locally parse + suite.

- [ ] **Step 1: Import the new helpers**

In the `using BiGSTARS: …` list, add `assemble_rows`, `_assemble_B_full`:
```julia
using BiGSTARS: _to_csr, _csr_row_block, _csr_block_nnz_split, _eps_options,
                _sigma_schedule, _group_indices, sparse_from_csr, assemble_rows,
                _assemble_B_full, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes
```
(`_csr_row_block`/`sparse_from_csr` may become unused after this task — leave them imported; harmless.)

- [ ] **Step 2: Replace `_build_petsc_mat` with a local builder**

Replace the entire `_build_petsc_mat` function with `_fill_mat!` + `_build_petsc_mats_local`:

```julia
"""Fill an already-created PETSc matrix `M` from a local owned-row slice `rows`
(`nrows × N`, global columns), owned global rows `[rstart,rend)`. No scatter."""
function _fill_mat!(M, rows, rstart::Integer, rend::Integer, comm::MPI.Comm)
    rowptr, colind, vals = _to_csr(rows)                       # CSR of the local slice
    d_nnz, o_nnz = _csr_block_nnz_split(rowptr, colind, 0, rend - rstart, rstart, rend)
    _prealloc!(M, MPI.Comm_size(comm), d_nnz, o_nnz) || MatSetUp(M)
    _insert_rows!(M, rstart, rowptr, colind, vals)            # global rows rstart..rend-1
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY)
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY)
    return M
end

"""Build distributed PETSc A and B for one wavenumber, each rank assembling only
its owned rows locally from the replicated cache (no rank-0 full matrix, no scatter)."""
function _build_petsc_mats_local(cache, k, N::Integer, comm::MPI.Comm)
    A = MatCreate(comm)
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(A)
    rstart, rend = MatGetOwnershipRange(A)                    # 0-based [rstart,rend)
    A_rows, B_rows = assemble_rows(cache, Float64(k), rstart, rend)
    _fill_mat!(A, A_rows, rstart, rend, comm)

    B = MatCreate(comm)
    MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, PetscInt(N), PetscInt(N))
    MatSetFromOptions(B)
    rstartB, rendB = MatGetOwnershipRange(B)
    @assert (rstartB, rendB) == (rstart, rend)               # same N+comm ⇒ same layout
    _fill_mat!(B, B_rows, rstart, rend, comm)
    return A, B
end
```

- [ ] **Step 3: Rewire `_solve_one_adaptive` to take `cache, k`**

Change its signature and the A,B build + the filter. Replace the head of
`_solve_one_adaptive` (the `function … ; A = _build_petsc_mat(...); B = _build_petsc_mat(...)` part) with:

```julia
function _solve_one_adaptive(cache, k, N::Integer, comm::MPI.Comm;
                             sigma_0, n_tries, Δσ₀, incre, ϵ, verbose)
    rank = MPI.Comm_rank(comm)
    t0 = MPI.Wtime()
    A, B = _build_petsc_mats_local(cache, k, N, comm)
    schedule = _sigma_schedule(sigma_0, n_tries, Δσ₀, incre)
```
And in the rank-0 filter line, replace `sparse_from_csr(B_csr)` with the locally
built mass matrix:
```julia
                λf, Χf = _filter_physical_modes(λ, Χ, _assemble_B_full(cache, Float64(k)))
```
(The σ-loop, gather, `EPSDestroy`, `MatDestroy(A)`/`MatDestroy(B)`, and the
`SolverResults` construction are unchanged.)

- [ ] **Step 4: Rewire `solve` — drop rank-0 assemble, scatter, and N-bcast**

In `BiGSTARS.solve`, replace the per-wavenumber loop body (the `A_csr=nothing … N=MPI.bcast(N,…) … _solve_one_adaptive(A_csr,B_csr,N,…)` block) with:

```julia
    for i in _group_indices(nk, ngroups, group_id)
        N = cache.N_total                          # replicated; no assemble-on-root, no bcast
        verbose && grank == 0 &&
            println("solve: group $(group_id)  k=$(k_values[i])  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")
        r = _solve_one_adaptive(cache, Float64(k_values[i]), N, group_comm;
                        sigma_0=Float64(sigma_0), n_tries=Int(n_tries),
                        Δσ₀=Float64(Δσ₀), incre=Float64(incre), ϵ=Float64(ϵ),
                        verbose=verbose)
        grank == 0 && push!(local_pairs, (i, r))
    end
```
(Everything else in `solve` — validation, group split, gather — is unchanged.)

- [ ] **Step 5: Parse-check + cross-platform suite**

Run:
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("ext parse OK")'
"$JL" --project=. -e 'using Pkg; Pkg.test()'
```
Expected: `ext parse OK`; suite PASS (ext not loaded in the test env; the
`assemble_rows`/helpers tests from T2/T3 cover the new logic locally).

- [ ] **Step 6: Commit**
```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: row-local distributed assembly in the solver (no rank-0 build, no scatter)"
```

Verification: **CI (mpi.yml)** n=1/2/4 — the existing analytic tests (`σ₁=k²+π²`,
residual, spurious filter, groups) must still pass, now produced via distributed
assembly. Debug insertion/ownership against CI if red.

---

## Task 5: Final verification

- [ ] **Step 1: Cross-platform suite + ext parse**
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" --project=. -e 'using Pkg; Pkg.test()'
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("OK")'
```
Expected: PASS; `OK`.

- [ ] **Step 2: Push + drive mpi.yml green** (commits/PR only on user authorization — auto-sync + no-commit rule). The n=1/2/4 analytic runs are the integration gate for the distributed assembly.

---

## Self-review

**Spec coverage:** `assemble_rows` pure core → T3; `_place_in_block_rows` → T2; `_k_values` (DRY, exact `assemble` match) → T1; distributed build / no-scatter / no rank-0 full A → T4 (`_build_petsc_mats_local`, `_fill_mat!`); `solve`/`_solve_one_adaptive` integration + drop N-bcast → T4; mass-filter full-B on root → T4 (`_assemble_B_full`, built in T3); cross-platform unit tests (plain/augmented/legacy, multiple ranges) → T3; MPI CI gate → T4/T5. Out-of-scope (2b) untouched. All covered.

**Placeholder scan:** none — full code per step, exact paths/commands.

**Type consistency:** `assemble_rows(cache, k::Float64, rstart, rend) -> (A_rows,B_rows)` defined T3, called T4 with `Float64(k)`; `_place_in_block_rows(mat, eq_idx, var_idx, N_per_var, N_vars, rstart, rend)` defined T2, called in T3 with matching arg order; `_k_values(cache, k)` defined T1, used T3; `_assemble_B_full(cache, k::Float64)` defined T3, called T4 with `Float64(k)`; `_fill_mat!`/`_build_petsc_mats_local` defined + called in T4; `_solve_one_adaptive(cache, k, N, comm; …)` new signature consistent between its definition (T4 s3) and call (T4 s4). `_to_csr`/`_csr_block_nnz_split`/`_prealloc!`/`_insert_rows!` reused with their existing signatures.
```
