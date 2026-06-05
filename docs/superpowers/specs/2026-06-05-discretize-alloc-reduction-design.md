# Design: discretize() allocation reduction — unify k-component storage, COO accumulation, drop dead serial-dense path

**Date:** 2026-06-05
**Status:** Approved (pending spec review)
**Scope (user-chosen):** Unify the duplicate k-component dicts into one source of truth, fix the sparse `+=` accumulation churn, AND remove the dead `assemble!`/workspace API.

## Motivation (measured)

`Profile.Allocs` of `discretize()` on a polar-vortex-shaped problem (5 vars, 4 derived → augmented, y-varying `Bz0`/`Bz0x`), two scales:

| | Nx=48 Nz=16 (N=6912) | Nx=96 Nz=24 (N=20736) |
|---|---|---|
| discretize total alloc | 1720 MB | 12769 MB |
| `_add_component!` (modern, KPowerKey) | 589 MB | 4211 MB |
| `_add_legacy_component!` (legacy, Int) | 589 MB | 4211 MB |
| **two together** | **68%** | **66%** |
| steady: legacy nnz / modern nnz | 1,064,165 / **1,064,165** | 7,271,774 / **7,271,774** |

Two findings:

1. **Byte-exact duplicate.** `A_components`/`B_components` (`Dict{Int,…}`, legacy) and `A_kcomponents`/`B_kcomponents` (`Dict{KPowerKey,…}`, modern) hold identical nnz. The legacy Int-keyed dict is read only by `assemble!` (in-place dense) — the path that fed the now-removed serial dense eigensolvers (Arpack/ArnoldiMethod/KrylovKit). The SLEPc path (`assemble_rows`) and serial `assemble`/`_assemble` use the modern KPowerKey dict.
2. **`+=` churn ≈ 35×.** Both `_add_*!` do `components[key] = components[key] + block`. A key's *final* sparse is 17 MB (Nx=48) but building it cost 589 MB — each addition reallocates the growing accumulator. Constant ~35× at both scales ⇒ at the real Nx=180 this is tens of GB of transient garbage and proportional GC time.

(The originally-suspected `_sparse_block_inverse` is **dead** in the augmented path: derived vars become regular variables, so `derived_caches` is empty and the function is never called during `discretize`.)

## Goal

Cut `discretize()` allocation ~3× (Nx=48: 1720→~560 MB; Nx=96: 12.8→~4.4 GB) and the matching GC time, with assembled `A,B` unchanged. No reduction of *peak* operator memory (that needs matrix-free operators — out of scope, established earlier).

## Design

### 1. Unify to one k-component dict

`DiscretizationCache` keeps a single KPowerKey-typed pair (rename the modern fields to the public names; drop the legacy Int pair):

```julia
struct DiscretizationCache
    A_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}   # was A_kcomponents
    B_components::Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}   # was B_kcomponents
    derived_caches::Dict{Symbol, DerivedVarCache}
    N_total::Int
    N_per_var::Int
    N_vars::Int
    domain::Domain
    derived_var_order::Vector{Symbol}
    row_range::Union{Nothing, Tuple{Int, Int}}
end
```

Delete: the Int-keyed `A_components`/`B_components` fields, `_add_legacy_component!`, `_legacy_components_to_k`, `_legacy_k_key`, and the outer Int-bridge constructor (`discretize.jl:64-75`). Replace the outer convenience constructor with a KPowerKey-typed one (same ergonomics, defaults `derived_var_order=Symbol[]`, `row_range=nothing`):

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

Rename `*_kcomponents` → `*_components` at every internal read:
- `discretize.jl`: `_assemble` (1916/1926), `_assembled_density` (2108), `Base.show` (166-171), build/BC sections (below).
- `mpi_prep.jl`: `assemble_rows` (265/271), `_assemble_B_full` (306), `restrict_cache_rows` (235-238).

`restrict_cache_rows` simplifies — it currently passes the legacy dict through **unsliced** (latent quirk, harmless because only the modern dict is consumed) and slices the modern one:
```julia
return DiscretizationCache(_slice_rows(cache.A_components), _slice_rows(cache.B_components),
    cache.derived_caches, cache.N_total, cache.N_per_var, cache.N_vars,
    cache.domain, cache.derived_var_order, (Int(rstart), Int(rend)))
```

### 2. COO-scatter accumulation (kills the churn and `place_in_block` in the build path)

Replace, in the build loop, `place_in_block(mat,…)` + `_add_*!(dict, key, block)` with offset-scatter of `mat`'s nonzeros into per-key triplet buffers, finalized once.

New helpers (`discretize.jl`, near the build loop):
```julia
const _COOBuf = Tuple{Vector{Int}, Vector{Int}, Vector{ComplexF64}}

function _scatter_block!(buffers::Dict{KPowerKey, _COOBuf}, key::KPowerKey,
                         mat::SparseMatrixCSC{ComplexF64, Int},
                         eq_idx::Int, var_idx::Int, N_per_var::Int)
    row_off = (eq_idx - 1) * N_per_var
    col_off = (var_idx - 1) * N_per_var
    I, J, V = get!(() -> (Int[], Int[], ComplexF64[]), buffers, key)
    rows = rowvals(mat); vals = nonzeros(mat)
    for col in 1:size(mat, 2)
        for idx in nzrange(mat, col)
            push!(I, rows[idx] + row_off); push!(J, col + col_off); push!(V, vals[idx])
        end
    end
    return buffers
end

function _finalize_components(buffers::Dict{KPowerKey, _COOBuf}, N_total::Int)
    out = Dict{KPowerKey, SparseMatrixCSC{ComplexF64, Int}}()
    for (key, (I, J, V)) in buffers
        out[key] = sparse(I, J, V, N_total, N_total)   # sums duplicate (i,j) — same as repeated +=
    end
    return out
end
```

Build loop (`discretize.jl` ~1640-1835) becomes:
- Init `A_buffers = Dict{KPowerKey,_COOBuf}()`, `B_buffers = Dict{KPowerKey,_COOBuf}()` (replacing the four dict inits at 1643-1646).
- RHS term: `mat = discretize_expr(...)`; `var_idx = find_target_variable(...)`; `_scatter_block!(A_buffers, kt.k_powers, mat, eq_idx, var_idx, N_per_var)`.
- LHS term: `_scatter_block!(B_buffers, kt.k_powers, mat, eq_idx, lhs_var_idx, N_per_var)`.
- After the equation loop: `A_components = _finalize_components(A_buffers, N_total)`; `B_components = _finalize_components(B_buffers, N_total)`.
- BC zeroing (1781-1800): keep only the `A_components`/`B_components` loops (the two `*_kcomponents` loops are gone).
- BC writes (1803-1833): keep only the KPowerKey (`kp`) branch — drop the Int `total_kp` branch and its `_total_k_power` use here. The `haskey`-create + scalar `+=` into the finalized sparse stays (small count; not a hotspot).
- Final `return DiscretizationCache(A_components, B_components, derived_caches, N_total, N_per_var, N_vars, prob.domain, derived_var_order, nothing)`.

`_add_component!` (KPowerKey, `discretize.jl:141`) **stays** — still used at 1701 to build a derived var's `op_k_components` (small, not a hotspot). `place_in_block` **stays** — still used by `_assemble` (1950) for derived terms.

Duplicate semantics: a key receiving contributions from multiple terms accumulates duplicate `(i,j)` triplets that `sparse()` sums — identical result to sequential `+=` (up to benign float summation order; operators are integer/rational-scaled, tests use tolerances).

### 3. Remove dead serial-dense API

Delete from `discretize.jl`: `assemble!` (2023-2093), `AssemblyWorkspace` (1994-1998), `allocate_workspace` (2007-2014). Remove their exports (`BiGSTARS.jl:31-33`). **Keep** `assemble`/`_assemble` (allocating serial pencil; used by the precompile workload at `BiGSTARS.jl:108` and many tests).

## Files touched

- `src/discretize.jl` — struct, constructors, build loop, BC sections, `_assemble`, `_assembled_density`, `Base.show`; delete legacy helpers + `assemble!`/workspace; add `_scatter_block!`/`_finalize_components`.
- `src/mpi_prep.jl` — `assemble_rows`, `_assemble_B_full`, `restrict_cache_rows` field renames + restrict simplification.
- `src/BiGSTARS.jl` — drop `assemble!`, `allocate_workspace`, `AssemblyWorkspace` exports.
- `test/test_discretize.jl` — imports (drop the 3 removed names); `cache.A_components` key checks Int→KPowerKey (`0`→`()`, `2`→`(:k_x=>2,)` per the test's `x=FourierTransformed` domain); `DiscretizationCache(...)` constructions (124/139/160/437) Int→KPowerKey dicts; delete the "In-place assembly (assemble!)" testset (363-386).
- `test/test_coverage_gaps.jl` — delete/port the `assemble!` derived-branch test (373).
- `docs/` — remove any `assemble!`/`allocate_workspace`/`AssemblyWorkspace` mentions (grep `docs/src` + `docs/src/modules/BiGSTARS.md`).

## Testing

- **Correctness (unchanged A,B):** full `Pkg.test()` — the existing `test_discretize.jl` assemble assertions exercise specific operator entries/structure at several k; they must still pass after the rename + COO change. Add one targeted test: two terms scattered to the same key sum correctly (`_finalize_components` == manual `+`).
- **Allocation regression guard:** add a test that discretizes a small problem and asserts `@allocated discretize(prob)` is below a generous bound that the old 35× churn would exceed (e.g. ≤ 6× the assembled steady nnz·16 bytes). Documents the win and prevents regression.
- **MPI:** `mpi.yml` n=1/2/4 — `assemble_rows`/`restrict_cache_rows`/`_assemble_B_full` read the renamed dict; the distributed solve + spurious filter must still match the analytic reference.
- **Manual:** re-run `/tmp/prof_bigstars.jl` (or equivalent) to confirm `discretize` total alloc dropped ~3× and `_add_legacy_component!` is gone from the profile.

## Risks

- **Float summation order** differs (`sparse()` dedup vs sequential `+`). Benign for these exactly-scaled spectral operators; tolerance-based tests cover it.
- **Public field rename** (`A_kcomponents`→`A_components`, type Int→KPowerKey) and **removed exports** are breaking — acceptable: v4.0.0 is already a breaking SLEPc-only release, and these served removed solvers. No deprecation shim (user chose full removal).
- **`push!`-grown COO buffers** reallocate (amortized 2×, not 35×). No `sizehint!` needed; could add later if profiling warrants.

## Out of scope

- `build_bc_rows` allocation (~127 MB at Nx=48 — the next hotspot after this) — separate effort.
- Peak operator memory (matrix-free) — separate architecture, established earlier.
- `_sparse_block_inverse` — dead in the augmented path; no change.
