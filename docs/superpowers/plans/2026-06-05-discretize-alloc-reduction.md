# discretize() Allocation Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut `discretize()` allocation ~3× by collapsing the duplicate k-component dicts into one KPowerKey source of truth, replacing `+=` sparse accumulation with COO-triplet accumulation, and deleting the dead serial-dense `assemble!`/workspace path.

**Architecture:** `DiscretizationCache` currently stores k-components twice — `A_components`/`B_components` (`Dict{Int,…}`, legacy, read only by the now-dead `assemble!`) and `A_kcomponents`/`B_kcomponents` (`Dict{KPowerKey,…}`, modern, read by SLEPc `assemble_rows` and serial `assemble`). The two are byte-identical. The build loop accumulates each via `components[key] = components[key] + block`, reallocating the growing accumulator on every term (~35× churn, measured). This plan: (1) add COO helpers, (2) delete `assemble!`/`AssemblyWorkspace`/`allocate_workspace`, (3) drop the Int dict and rename the KPowerKey dict to the public name, building it via COO scatter, (4) add an allocation-regression guard.

**Tech Stack:** Julia 1.12.x, SparseArrays (`sparse(I,J,V,m,n)` sums duplicate entries), Test.jl. Spec: `docs/superpowers/specs/2026-06-05-discretize-alloc-reduction-design.md`.

**Running Julia/tests in this repo** (juliaup + parts of `~/.julia` are root-owned):
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
$JL --project=. -e 'using Pkg; Pkg.test()'          # full suite (reliable)
```
Bash tool: run with `dangerouslyDisableSandbox: true`. The extension (`BiGSTARSMPIExt`) needs MPI/PETSc and is NOT loaded by `Pkg.test()`; the MPI path is verified by `.github/workflows/mpi.yml`, not locally.

---

## Key types (already defined — do not redefine)

- `const KPowerKey = Tuple{Vararg{Pair{Symbol, Int}}}` (`src/k_separation.jl:5`). Examples: `()` for k⁰, `(:k_x => 2,)` for k² in a domain with `x = FourierTransformed()`.
- `KTerm` has fields `.k_powers::KPowerKey` and `.k_power::Int` (total power). After this plan, only `.k_powers` is used in the build loop.

---

### Task 1: COO accumulation helpers

**Files:**
- Modify: `src/discretize.jl` (add helpers next to `_add_component!`, after line 161)
- Test: `test/test_discretize.jl` (add a testset)

These helpers are additive — the package stays green and existing behavior is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `test/test_discretize.jl` (inside the existing top-level `@testset`, e.g. after the first testset). It uses internal symbols via the `BiGSTARS.` prefix:

```julia
@testset "COO accumulation helpers" begin
    N_per_var = 3; N_total = 6
    buffers = Dict{BiGSTARS.KPowerKey, BiGSTARS._COOBuf}()

    # Two blocks at the SAME key and SAME block position (eq 1, var 1) must SUM.
    m1 = sparse([1, 2], [1, 2], ComplexF64[1.0, 2.0], 3, 3)
    m2 = sparse([1],    [1],    ComplexF64[10.0],      3, 3)
    BiGSTARS._scatter_block!(buffers, (), m1, 1, 1, N_per_var)
    BiGSTARS._scatter_block!(buffers, (), m2, 1, 1, N_per_var)

    # A different key at a different block position (eq 2, var 2).
    m3 = sparse([1], [2], ComplexF64[5.0], 3, 3)
    BiGSTARS._scatter_block!(buffers, (:k_x => 2,), m3, 2, 2, N_per_var)

    comps = BiGSTARS._finalize_components(buffers, N_total)

    A0 = comps[()]
    @test size(A0) == (6, 6)
    @test A0[1, 1] == ComplexF64(11.0)   # 1.0 + 10.0 summed (duplicate (i,j))
    @test A0[2, 2] == ComplexF64(2.0)
    @test nnz(A0) == 2

    A2 = comps[(:k_x => 2,)]
    @test A2[1 + 3, 2 + 3] == ComplexF64(5.0)   # offset by (eq2,var2) = (+3,+3)
    @test nnz(A2) == 1
end
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
Expected: FAIL — `UndefVarError: _COOBuf` / `_scatter_block!` not defined.

- [ ] **Step 3: Implement the helpers**

In `src/discretize.jl`, immediately after `_add_legacy_component!` (after line 161), add:

```julia
# COO-triplet accumulator for building a k-power component without repeated
# sparse `+=` (which reallocates the whole accumulator each term). Blocks are
# scattered into per-key (I, J, V) buffers and finalized with one `sparse()`,
# which sums duplicate (i, j) entries — identical result to sequential `+=`.
const _COOBuf = Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}

function _scatter_block!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                         mat::SparseMatrixCSC{ComplexF64, Int},
                         eq_idx::Int, var_idx::Int, N_per_var::Int)
    row_off = (eq_idx - 1) * N_per_var
    col_off = (var_idx - 1) * N_per_var
    I, J, V = get!(() -> (Int[], Int[], ComplexF64[]), buffers, key)
    rows = rowvals(mat)
    vals = nonzeros(mat)
    for col in 1:size(mat, 2)
        for idx in nzrange(mat, col)
            push!(I, rows[idx] + row_off)
            push!(J, col + col_off)
            push!(V, vals[idx])
        end
    end
    return buffers
end

function _finalize_components(buffers::Dict{KPowerKey, _COOBuf}, N_total::Int)
    out = Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}()
    for (key, (I, J, V)) in buffers
        out[key] = sparse(I, J, V, N_total, N_total)
    end
    return out
end
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS (full suite still green; new testset passes).

- [ ] **Step 5: Commit**

```bash
git add src/discretize.jl test/test_discretize.jl
git commit -m "feat: add COO-triplet accumulation helpers for discretize"
```

---

### Task 2: Remove the dead serial-dense API (`assemble!`, `AssemblyWorkspace`, `allocate_workspace`)

**Files:**
- Modify: `src/discretize.jl:1989-2093` (delete `AssemblyWorkspace`, `allocate_workspace`, `assemble!`)
- Modify: `src/BiGSTARS.jl:31-33` (remove three exports)
- Modify: `test/test_discretize.jl` (imports line 5-6; delete the `assemble!` testset ~363-386)
- Modify: `test/test_coverage_gaps.jl` (delete/port the `assemble!` derived-branch test ~373)

These three feed the removed serial dense eigensolvers. `assemble!` is the last functional reader of the legacy Int dict — removing it now keeps Task 3 green. The allocating `assemble`/`_assemble` is KEPT (precompile + tests use it).

- [ ] **Step 1: Delete the dead definitions in `src/discretize.jl`**

Remove the entire block from the `# In-place assembly workspace` section header through the end of `assemble!`. Concretely, delete lines from `1985` (the `# ───…` separator above `#  In-place assembly workspace`) through the end of the `assemble!` function (the `return ws` + `end` at ~2093). Leave the `_assembled_density` function (starts ~2095) intact.

Verify nothing else references the removed names:
```bash
grep -rnE 'assemble!|AssemblyWorkspace|allocate_workspace' src/
```
Expected: only the export lines in `src/BiGSTARS.jl` remain (removed next step).

- [ ] **Step 2: Remove the exports in `src/BiGSTARS.jl`**

Delete these three lines (currently 31-33):
```julia
        assemble!,
        allocate_workspace,
        AssemblyWorkspace,
```
Keep the `assemble,` export (line 30).

- [ ] **Step 3: Remove the tests that exercise the deleted API**

In `test/test_discretize.jl`:
- Line 5: change `total_grid_size, discretize, assemble, allocate_workspace, assemble!,` → `total_grid_size, discretize, assemble,`
- Line 6: remove `AssemblyWorkspace, ` from the import list (keep `DiscretizationCache`, `ParamNode`, etc.).
- Delete the whole `@testset "In-place assembly (assemble!)" begin … end` block (~363-386).

In `test/test_coverage_gaps.jl`: locate the test around line 373 that calls `assemble!` (comment mentions "in-place assemble! (assemble! + H(k) reconstruction)") and delete that `@testset`/block. Verify:
```bash
grep -rnE 'assemble!|AssemblyWorkspace|allocate_workspace' test/
```
Expected: no matches.

- [ ] **Step 4: Run the suite**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS. (The legacy Int dict is now built + shown but has no functional reader — removed in Task 3.)

- [ ] **Step 5: Commit**

```bash
git add src/discretize.jl src/BiGSTARS.jl test/test_discretize.jl test/test_coverage_gaps.jl
git commit -m "refactor: remove dead serial-dense assemble!/AssemblyWorkspace/allocate_workspace"
```

---

### Task 3: Unify to one KPowerKey dict, build it via COO scatter

**Files:**
- Modify: `src/discretize.jl` — struct (24-36), constructors (64-94), delete `_add_legacy_component!` (152-161), build loop (1643-1836), `_assemble` (1916/1926), `_assembled_density` (~2108), `Base.show` (166-171)
- Modify: `src/mpi_prep.jl` — `restrict_cache_rows` (225-239), `assemble_rows` (265/271), `_assemble_B_full` (306)
- Modify: `test/test_discretize.jl` — field-key checks (43-47), `DiscretizationCache(...)` constructions (124/139/160/437)
- Modify: `test/test_mpi_prep.jl` — any `*_kcomponents`/`A_components` references (grep)

This is one compile-coherent change (renaming a struct field breaks the package until all readers update). The existing `assemble`/`assemble_rows` tests are the correctness net: assembled `A,B` must be unchanged.

- [ ] **Step 1: Replace the struct definition** (`src/discretize.jl:24-36`)

```julia
"""Cache of pre-discretized sparse matrix components, separated by k-power."""
struct DiscretizationCache
    A_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}
    B_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}
    derived_caches::Dict{Symbol, DerivedVarCache}
    N_total::Int
    N_per_var::Int
    N_vars::Int
    domain::Domain
    derived_var_order::Vector{Symbol}
    row_range::Union{Nothing, Tuple{Int, Int}}
end
```

- [ ] **Step 2: Replace the outer convenience constructor and delete legacy key helpers** (`src/discretize.jl:64-94`)

Replace the `DiscretizationCache(A_components::Dict{Int,…}, …)` constructor (64-75) AND delete `_legacy_components_to_k` (77-84) and `_legacy_k_key` (86-94) with a single KPowerKey convenience constructor:

```julia
function DiscretizationCache(A_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                             B_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}},
                             derived_caches::Dict{Symbol, DerivedVarCache},
                             N_total::Int, N_per_var::Int, N_vars::Int, domain::Domain,
                             derived_var_order::Vector{Symbol}=Symbol[])
    return DiscretizationCache(A_components, B_components, derived_caches,
                               N_total, N_per_var, N_vars, domain, derived_var_order, nothing)
end
```

Keep `_total_k_power` (96) and `_k_coeff` (98) — they operate on `KPowerKey` and are still used.

- [ ] **Step 3: Delete `_add_legacy_component!`** (`src/discretize.jl:152-161`)

Remove the whole function. (`_add_component!` at 141 STAYS — still used at line ~1701 for derived `op_k_components`.)

- [ ] **Step 4: Update `Base.show`** (`src/discretize.jl:166-171`)

Change `c.A_components`/`c.B_components` reads — they are now KPowerKey-keyed, which `sort` cannot order directly. Replace lines 166-171 with:

```julia
    a_keys = collect(keys(c.A_components))
    b_keys = collect(keys(c.B_components))
    println(io, "  A components (", length(a_keys), " k-power keys)")
    println(io, "  B components (", length(b_keys), " k-power keys)")
    total_nnz = sum(nnz(v) for v in values(c.A_components)) +
                sum(nnz(v) for v in values(c.B_components))
```

- [ ] **Step 5: Rewrite the build-loop accumulator init** (`src/discretize.jl:1643-1646`)

Replace the four dict inits with two COO buffer dicts:

```julia
    A_buffers = Dict{KPowerKey, _COOBuf}()
    B_buffers = Dict{KPowerKey, _COOBuf}()
```

- [ ] **Step 6: Replace the RHS per-term accumulation** (`src/discretize.jl:1738-1743`)

Current:
```julia
            mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
            var_idx = find_target_variable(kt.expr, prob.variables)
            block = place_in_block(mat, eq_idx, var_idx, N_vars, N_per_var)

            _add_legacy_component!(A_components, kt.k_power, block)
            _add_component!(A_kcomponents, kt.k_powers, block)
```
Replace with:
```julia
            mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
            var_idx = find_target_variable(kt.expr, prob.variables)
            _scatter_block!(A_buffers, kt.k_powers, mat, eq_idx, var_idx, N_per_var)
```

- [ ] **Step 7: Replace the LHS per-term accumulation** (`src/discretize.jl:1755-1761`)

Current:
```julia
            for kt in lhs_terms
                mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
                b_block = place_in_block(mat, eq_idx, lhs_var_idx, N_vars, N_per_var)

                _add_legacy_component!(B_components, kt.k_power, b_block)
                _add_component!(B_kcomponents, kt.k_powers, b_block)
            end
```
Replace with:
```julia
            for kt in lhs_terms
                mat = discretize_expr(kt.expr, prob, N_per_var, eq_order, ctx)
                _scatter_block!(B_buffers, kt.k_powers, mat, eq_idx, lhs_var_idx, N_per_var)
            end
```

- [ ] **Step 8: Finalize buffers + rewrite the BC sections** (`src/discretize.jl:1766-1836`)

After the `end` that closes the `for (eq_idx, eq)` equation loop (line 1764) and before `# Apply BCs`, insert:

```julia
    A_components = _finalize_components(A_buffers, N_total)
    B_components = _finalize_components(B_buffers, N_total)
```

Replace the four BC-zeroing loops (1781-1800) with two (single dict each):

```julia
    for p in keys(A_components)
        for row_idx in bc_row_indices
            A_components[p][row_idx, :] .= zero(ComplexF64)
        end
    end
    for p in keys(B_components)
        for row_idx in bc_row_indices
            B_components[p][row_idx, :] .= zero(ComplexF64)
        end
    end
```

Replace the BC-row write block (1803-1833) with the KPowerKey-only version:

```julia
    for (row_idx, kp, a_row, b_row) in bc_info
        if !haskey(A_components, kp)
            A_components[kp] = spzeros(ComplexF64, N_total, N_total)
        end
        if !haskey(B_components, kp)
            B_components[kp] = spzeros(ComplexF64, N_total, N_total)
        end
        for (j, v) in enumerate(a_row)
            v != 0.0 && (A_components[kp][row_idx, j] += ComplexF64(v))
        end
        for (j, v) in enumerate(b_row)
            v != 0.0 && (B_components[kp][row_idx, j] += ComplexF64(v))
        end
    end
```

Replace the final return (1835-1836):

```julia
    return DiscretizationCache(A_components, B_components, derived_caches,
                               N_total, N_per_var, N_vars, prob.domain,
                               derived_var_order, nothing)
```

- [ ] **Step 9: Rename reads in `_assemble`** (`src/discretize.jl:1916, 1926`)

Change `cache.A_kcomponents` → `cache.A_components` (line 1916) and `cache.B_kcomponents` → `cache.B_components` (line 1926). Leave the rest of `_assemble` unchanged.

- [ ] **Step 10: Rename read in `_assembled_density`** (`src/discretize.jl:~2108`)

Change `for (_, M) in cache.A_kcomponents` → `for (_, M) in cache.A_components`.

- [ ] **Step 11: Update `src/mpi_prep.jl`**

`restrict_cache_rows` (225-239) — slice the now-single dict pair and drop the legacy pass-through. Replace the `return DiscretizationCache(...)` (235-238) with:
```julia
    return DiscretizationCache(_slice_rows(cache.A_components), _slice_rows(cache.B_components),
        cache.derived_caches, cache.N_total, cache.N_per_var, cache.N_vars,
        cache.domain, cache.derived_var_order, (Int(rstart), Int(rend)))
```
Also update the docstring reference `A_kcomponents`/`B_kcomponents` (line 220) → `A_components`/`B_components`.

`assemble_rows` (265, 271) — `cache.A_kcomponents` → `cache.A_components`, `cache.B_kcomponents` → `cache.B_components`.

`_assemble_B_full` (306) — `cache.B_kcomponents` → `cache.B_components`. Update its docstring (298) too.

- [ ] **Step 12: Update `test/test_discretize.jl` field-key checks** (43-47)

The dict is now KPowerKey-keyed. For a domain with `x = FourierTransformed()`, k⁰ is `()` and k² is `(:k_x => 2,)`. Replace:
```julia
        @test haskey(cache.A_components, 0)
        @test haskey(cache.A_components, 2)
```
with:
```julia
        @test haskey(cache.A_components, ())
        @test haskey(cache.A_components, (:k_x => 2,))
```
and likewise the `B_components` check at line 47 (`0` → `()`). **First confirm the transformed dim name** for that testset's domain:
```bash
grep -nE 'Domain\(|FourierTransformed|Fourier\(|Chebyshev\(' test/test_discretize.jl | sed -n '1,20p'
```
If the transformed coord is not `x`, use `(:k_<name> => 2,)` accordingly.

- [ ] **Step 13: Update direct `DiscretizationCache(...)` constructions in tests** (124, 139, 160, 437)

These build caches with Int-keyed dicts (e.g. `A0 = Dict(0 => …, 2 => …)`). Convert each to KPowerKey keys: `0` → `()`, `2` → `(:k_x => 2,)` (use the actual transformed-dim name for that test's domain). Example for line 437's `DiscretizationCache(A0, B0, dc, 8, 8, 1, domain)`: change the construction of `A0`/`B0` to use KPowerKey keys; the 8-arg call shape still matches the new convenience constructor. Read each site first:
```bash
sed -n '118,168p;430,445p' test/test_discretize.jl
```
Update the dict literals to KPowerKey keys. The constructor arity is unchanged (the convenience ctor still takes `A, B, derived_caches, N_total, N_per_var, N_vars, domain[, derived_var_order]`).

- [ ] **Step 14: Catch any remaining `*_kcomponents` references**

```bash
grep -rnE 'A_kcomponents|B_kcomponents' src/ ext/ test/
```
Expected: no matches. Fix any stragglers (e.g. in `test/test_mpi_prep.jl`) by renaming to `A_components`/`B_components`. (The extension consumes the cache only through `assemble_rows`/`restrict_cache_rows`/`_assemble_B_full`, whose signatures are unchanged — but grep `ext/` anyway to be sure.)

- [ ] **Step 15: Run the suite**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS — same count as before Task 3 (assembled `A,B` unchanged; COO `sparse()` sums == `+=`).

- [ ] **Step 16: Commit**

```bash
git add src/discretize.jl src/mpi_prep.jl test/test_discretize.jl test/test_mpi_prep.jl
git commit -m "refactor: unify k-component storage to single KPowerKey dict via COO scatter"
```

---

### Task 4: Allocation-regression guard + docs sweep

**Files:**
- Modify: `test/test_discretize.jl` (add an allocation testset)
- Modify: docs under `docs/src/` (remove mentions of the deleted API)

- [ ] **Step 1: Write the allocation-regression test**

Add to `test/test_discretize.jl` (top-level testset). It builds a small multi-term 2D problem, warms up compilation, then asserts `discretize` allocation is within a generous multiple of the assembled steady size (the old ~35× churn would blow past it):

```julia
@testset "discretize allocation stays bounded (no += churn)" begin
    dom = Domain(x = FourierTransformed(),
                 y = Fourier(12, [0.0, 2π]),
                 z = Chebyshev(8, [0.0, 1.0]))
    prob = EVP(dom, variables = [:w, :b], eigenvalue = :Ra)
    Y, Z = meshgrid(dom, :y, :z)
    prob[:F] = vec(@. exp(-((Y - π)^2)))          # y-varying field → multiple terms/key
    @substitution D2(A) = dx(dx(A)) + dy(dy(A)) + dz(dz(A))
    @equation Ra * w = D2(w) + F * dy(b)
    @equation 0 = b + D2(b) - F * w
    @bc left(w) = 0; @bc right(w) = 0
    @bc left(b) = 0; @bc right(b) = 0

    discretize(prob)                               # warm up (compile)
    bytes  = @allocated discretize(prob)
    cache  = discretize(prob)
    steady = (sum(nnz, values(cache.A_components)) +
              sum(nnz, values(cache.B_components))) * 16   # ComplexF64 = 16 bytes
    @test bytes < 10 * steady                      # COO ~2×; old churn was ~35×
end
```

If `steady` is tiny for this case and `10×` proves too tight on the green build, raise the constant and add a comment with the measured ratio — the point is to catch a regression to tens-of-× churn, not to pin an exact number.

- [ ] **Step 2: Run the test**

```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS. Note the printed test count.

- [ ] **Step 3: Docs sweep**

```bash
grep -rnE 'assemble!|allocate_workspace|AssemblyWorkspace|A_kcomponents|B_kcomponents' docs/
```
For each hit in `docs/src/**` (e.g. `docs/src/modules/BiGSTARS.md`, performance/method pages), remove the reference or rewrite it to the current API (`assemble`, single `A_components`/`B_components`). If a doc lists exported symbols, drop the three removed names.

- [ ] **Step 4: Commit**

```bash
git add test/test_discretize.jl docs/
git commit -m "test: guard discretize allocation; docs: drop removed assemble! API"
```

---

### Final verification (after all tasks)

- [ ] Full suite green:
```bash
$JL --project=. -e 'using Pkg; Pkg.test()'
```
- [ ] Extension still parses (MPI path is CI-verified, not run locally):
```bash
$JL --project=. -e 'using BiGSTARS; include("ext/BiGSTARSMPIExt.jl")' 2>&1 | head
```
Expected: parses without `UndefVarError` for renamed symbols (it imports `assemble_rows`, `restrict_cache_rows`, `_assemble_B_full` — all updated). A load error from missing PetscWrap/SlepcWrap symbols is expected and fine; a `*_kcomponents` `UndefVarError` is NOT.
- [ ] No stragglers:
```bash
grep -rnE 'A_kcomponents|B_kcomponents|_add_legacy_component|_legacy_components_to_k|_legacy_k_key|assemble!|AssemblyWorkspace|allocate_workspace' src/ ext/ test/
```
Expected: no matches.
- [ ] (Optional) Re-run the allocation profile to confirm the ~3× drop and that `_add_legacy_component!` is gone — see `/tmp/prof_bigstars.jl`.
- [ ] Hand off to **superpowers:finishing-a-development-branch** (present merge/PR options; no push to `main` or PR until the user chooses; this branch shares `mpi_prep.jl` edits with the open 2b-ii PR #84, so watch for conflict at merge).
