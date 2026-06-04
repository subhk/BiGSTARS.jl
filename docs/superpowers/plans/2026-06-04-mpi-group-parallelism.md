# MPI group parallelism (across-wavenumber) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the eigensolve across wavenumbers by splitting MPI ranks into groups, each group running the existing distributed solve on its own sub-communicator for a round-robin subset of `k_values`.

**Architecture:** Add an `ngroups` kwarg to `BiGSTARS.solve`. Split `COMM_WORLD` into `ngroups` equal groups via `MPI.Comm_split`; group `g` owns `k_values[i]` where `(i-1) % ngroups == g`; each group runs the existing `_solve_one_adaptive` on its `group_comm`; group roots ship results to global rank 0. `ngroups=1` reproduces the v4.0.0 single-communicator behavior exactly.

**Tech Stack:** Julia, MPI.jl 0.19 (positional-root collectives + `Comm_split`), PetscWrap/SlepcWrap 0.1.x (comm-parameterized `MatCreate`/`EPSCreate`), complex PETSc/SLEPc (solve verified in CI only).

---

## Verification reality

- `_group_indices` is pure Julia → unit-tested locally + in the cross-platform CI matrix.
- Everything touching MPI/PETSc (the `solve` rewrite, the group solve) is **CI-only** (`mpi.yml`); locally only parse-check + the cross-platform suite. Run Julia via the project's invocation (juliaup binary + `JULIA_DEPOT_PATH=/tmp/jdepot:~/.julia`, unsandboxed).

## File structure

```
src/mpi_prep.jl              modify: + _group_indices (pure, round-robin partition)
src/BiGSTARS.jl              (no change — _group_indices is internal, used by the ext)
ext/BiGSTARSMPIExt.jl        modify: MatCreate(comm)/EPSCreate(comm); rewrite BiGSTARS.solve top-level with ngroups + group split + cross-group gather; import _group_indices
test/test_mpi_prep.jl        modify: + _group_indices unit tests (cross-platform)
test/mpi/test_slepc.jl       modify: + across-k group section (adaptive ngroups)
.github/workflows/mpi.yml    modify: + an `mpiexec -n 4` run
docs/src/mpi.md              modify: document ngroups
```

---

## Task 1: `_group_indices` round-robin partition helper

**Files:**
- Modify: `src/mpi_prep.jl`
- Test: `test/test_mpi_prep.jl`

- [ ] **Step 1: Write failing tests**

Append inside the top-level `@testset` in `test/test_mpi_prep.jl`:

```julia
@testset "_group_indices round-robin partition" begin
    # 5 wavenumbers, 2 groups
    @test BiGSTARS._group_indices(5, 2, 0) == [1, 3, 5]
    @test BiGSTARS._group_indices(5, 2, 1) == [2, 4]
    # 1 group gets everything
    @test BiGSTARS._group_indices(4, 1, 0) == [1, 2, 3, 4]
    # every index assigned exactly once, across all groups (coverage + disjoint)
    let nk = 7, ng = 3
        all_idx = reduce(vcat, BiGSTARS._group_indices(nk, ng, g) for g in 0:(ng-1))
        @test sort(all_idx) == collect(1:nk)
    end
    # empty group (more groups than wavenumbers)
    @test BiGSTARS._group_indices(2, 4, 3) == Int[]
end
```

- [ ] **Step 2: Run tests, verify they fail**

Run:
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" --project=. test/test_mpi_prep.jl
```
Expected: FAIL — `_group_indices` undefined (`UndefVarError`).

- [ ] **Step 3: Add `_group_indices` to `src/mpi_prep.jl`**

Insert after the `_sigma_schedule` function:

```julia
"""
    _group_indices(nk, ngroups, group_id) -> Vector{Int}

1-based indices of `1:nk` assigned to group `group_id` (0-based) under a
round-robin split into `ngroups` groups: index `i` goes to group `(i-1) % ngroups`.
Pure-Julia, so the across-wavenumber routing is unit-tested without MPI. Returns
an empty vector when a group gets no work (more groups than wavenumbers).
"""
function _group_indices(nk::Integer, ngroups::Integer, group_id::Integer)
    return Int[i for i in 1:nk if (i - 1) % ngroups == group_id]
end
```

- [ ] **Step 4: Run tests, verify pass**

Run: `"$JL" --project=. test/test_mpi_prep.jl`
Expected: PASS (all `_group_indices` asserts green; rest of file still passes).

- [ ] **Step 5: Commit**

```bash
git add src/mpi_prep.jl test/test_mpi_prep.jl
git commit -m "feat: add _group_indices round-robin wavenumber partition helper"
```

---

## Task 2: `ngroups` in `BiGSTARS.solve` + comm-parameterized solve path

**Files:**
- Modify: `ext/BiGSTARSMPIExt.jl`

Verification is **CI-only** (needs MPI + complex PETSc). Locally: confirm the file
parses (Step 5).

- [ ] **Step 1: Pass the communicator to `MatCreate`**

In `ext/BiGSTARSMPIExt.jl`, in `_build_petsc_mat`, change the matrix creation line:

```julia
    M = MatCreate()                                # defaults to MPI.COMM_WORLD
```
to:
```julia
    M = MatCreate(comm)                            # create on the (group) communicator
```

- [ ] **Step 2: Pass the communicator to `EPSCreate`**

In `_solve_one_adaptive`, change:

```julia
        eps = EPSCreate()
```
to:
```julia
        eps = EPSCreate(comm)
```

(The rest of `_build_petsc_mat`, `_gather_eigenpairs`, and `_solve_one_adaptive`
already use the passed `comm` for every `MPI.Gather`/`Allgather`/`send`/`recv`/
`bcast`/`Gatherv!`, and `MatCreateVecs`/`MatGetOwnershipRange` follow the matrix's
comm. So once `MatCreate`/`EPSCreate` take `comm`, the whole per-pencil solve runs
on `comm`.)

- [ ] **Step 3: Import `_group_indices`**

In the `using BiGSTARS: …` list at the top of the module, add `_group_indices`:

```julia
using BiGSTARS: _to_csr, _csr_row_block, _csr_block_nnz_split, _eps_options,
                _sigma_schedule, _group_indices, sparse_from_csr, SolverResults,
                ConvergenceHistory, sort_eigenvalues!, _filter_physical_modes
```

- [ ] **Step 4: Rewrite the public `solve(cache, k_values; …)` method**

Replace the entire `function BiGSTARS.solve(cache::BiGSTARS.DiscretizationCache, k_values::AbstractVector; …) … end` method (the multi-arg one) with:

```julia
function BiGSTARS.solve(cache::BiGSTARS.DiscretizationCache,
                        k_values::AbstractVector;
                        sigma_0::Real, nev::Integer=1, which::Symbol=:LM,
                        tol::Real=1e-10, maxiter::Integer=300, ncv::Integer=0,
                        mat_solver::Symbol=:mumps, eps_type::Symbol=:krylovschur,
                        n_tries::Integer=8, Δσ₀::Real=0.2, incre::Real=1.2, ϵ::Real=1e-5,
                        ngroups::Integer=1, manage_init::Bool=true, verbose::Bool=false)
    opts = _eps_options(; nev=Int(nev), which=which, tol=Float64(tol),
                        maxiter=Int(maxiter), ncv=Int(ncv),
                        mat_solver=String(mat_solver), eps_type=String(eps_type))

    # MPI.jl populates COMM_WORLD only through its own MPI.Init(); the C-level
    # MPI_Init PETSc runs inside SlepcInitialize does NOT. Init MPI.jl first, then
    # let PETSc reuse the already-initialized MPI.
    MPI.Initialized() || MPI.Init()
    if manage_init
        if !_SLEPC_INITED[]
            SlepcInitialize(opts)
            _SLEPC_INITED[] = true
            _SLEPC_OPTS[] = opts
        elseif opts != _SLEPC_OPTS[]
            error("solve: SLEPc is already initialized in this process with " *
                  "different options, and PETSc/SLEPc options can only be set once " *
                  "per process. Restart Julia for new solver settings, or pass " *
                  "manage_init=false and drive SlepcInitialize yourself.\n" *
                  "  initialized with: $(_SLEPC_OPTS[])\n  requested now:    $(opts)")
        end
    end
    _assert_complex_scalars()

    world = MPI.COMM_WORLD
    P = MPI.Comm_size(world)
    wrank = MPI.Comm_rank(world)
    # Validate identically on every rank (pre-split, so a bad ngroups errors
    # collectively — no deadlock).
    (1 ≤ ngroups ≤ P) ||
        error("solve: ngroups=$(ngroups) must be in 1:$(P) (number of MPI ranks)")
    (P % ngroups == 0) ||
        error("solve: nprocs=$(P) is not divisible by ngroups=$(ngroups)")

    nk = length(k_values)
    # Full-length result vector; non-(global-root) ranks return empty markers.
    results = SolverResults[
        SolverResults(ComplexF64[], zeros(ComplexF64, 0, 0), false, :Slepc,
                      Float64(sigma_0), 0, 0.0, ConvergenceHistory()) for _ in 1:nk]

    # Form groups. ngroups==1 keeps COMM_WORLD verbatim (the v4.0.0 path).
    if ngroups == 1
        group_comm = world; group_id = 0; psize = P
    else
        psize = P ÷ ngroups
        group_id = wrank ÷ psize
        group_comm = MPI.Comm_split(world, group_id, wrank)
    end
    grank = MPI.Comm_rank(group_comm)

    # Each group solves its round-robin subset, sequentially, distributed on group_comm.
    local_pairs = Tuple{Int,SolverResults}[]
    for i in _group_indices(nk, ngroups, group_id)
        A_csr = nothing; B_csr = nothing; N = 0
        if grank == 0
            A, B = BiGSTARS.assemble(cache, Float64(k_values[i]))
            N = size(A, 1)
            A_csr = _to_csr(A)
            B_csr = _to_csr(B)
        end
        N = MPI.bcast(N, 0, group_comm)            # group-collective
        verbose && grank == 0 &&
            println("solve: group $(group_id)  k=$(k_values[i])  N=$(N)  σ₀=$(sigma_0)  nev=$(nev)")
        r = _solve_one_adaptive(A_csr, B_csr, N, group_comm;
                        sigma_0=Float64(sigma_0), n_tries=Int(n_tries),
                        Δσ₀=Float64(Δσ₀), incre=Float64(incre), ϵ=Float64(ϵ),
                        verbose=verbose)
        grank == 0 && push!(local_pairs, (i, r))
    end

    # Collect group-root results to global rank 0 (point-to-point over WORLD).
    if ngroups == 1
        wrank == 0 && for (i, r) in local_pairs
            results[i] = r
        end
    else
        TAG = 7777
        if wrank == 0
            for (i, r) in local_pairs                 # own group's results
                results[i] = r
            end
            for g in 1:(ngroups - 1)                   # the other groups' roots
                pairs, _ = MPI.recv(g * psize, TAG, world)
                for (i, r) in pairs
                    results[i] = r
                end
            end
        elseif grank == 0
            MPI.send(local_pairs, 0, TAG, world)       # group root → global root
        end
    end

    return results
end
```

- [ ] **Step 5: Parse-check locally**

Run:
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("ext parse OK")'
```
Expected: `ext parse OK`. (The ext cannot be loaded/run without MPI+complex PETSc — runtime verification is the `mpi.yml` job in Task 3.)

- [ ] **Step 6: Cross-platform suite still green (ext not loaded there)**

Run: `"$JL" --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (the extension is not triggered in the test env; this confirms the `_group_indices` import line and the package still load cleanly).

- [ ] **Step 7: Commit**

```bash
git add ext/BiGSTARSMPIExt.jl
git commit -m "feat: ngroups across-wavenumber parallelism via MPI sub-communicators"
```

---

## Task 3: MPI CI test for groups

**Files:**
- Modify: `test/mpi/test_slepc.jl`, `.github/workflows/mpi.yml`

Verification is **CI-only**.

- [ ] **Step 1: Add the across-k group section to `test/mpi/test_slepc.jl`**

Append at the end of `test/mpi/test_slepc.jl` (after the spurious-filter section).
It reuses the Poisson `cache` and the SAME static options (`nev=4, which=:LM,
tol=1e-10`) as the earlier solves — required, since SLEPc options are fixed once
per process. `ngroups` is adaptive so the script is valid at `-n 1/2/4`:

```julia
# ── Across-wavenumber groups (Phase 1) ────────────────────────────────────────
# Split ranks into 2 groups when nprocs is even (n=2 → 2×1, n=4 → 2×2); each group
# solves a round-robin subset of the wavenumbers and global rank 0 collects all.
# σ_1(k) = k² + π² for this Poisson pencil.
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
ng = (nprocs % 2 == 0) ? 2 : 1
ks = [0.5, 1.0, 1.5, 2.0]
res_g = solve(cache, ks; sigma_0=10.0, nev=4, which=:LM, tol=1e-10, ngroups=ng)
if rank == 0
    ts3 = @testset "across-k groups (ngroups=$(ng))" begin
        @test length(res_g) == length(ks)
        for (j, kj) in enumerate(ks)
            @test res_g[j].converged
            σ1 = minimum(real, res_g[j].eigenvalues)   # smallest = k² + π²
            @test isapprox(σ1, kj^2 + π^2; rtol=1e-3)
        end
        println("groups ng=$(ng): " *
                join(["k=$(ks[j]) σ1=$(minimum(real,res_g[j].eigenvalues))" for j in 1:length(ks)], "  "))
    end
    ts3.anynonpass && exit(1)
end
```

- [ ] **Step 2: Add an `-n 4` run to `.github/workflows/mpi.yml`**

After the existing `Run integration test (n=2)` step, add:

```yaml
      - name: Run integration test (n=4, groups)
        run: mpiexec --oversubscribe -n 4 julia --project=test/mpi test/mpi/test_slepc.jl
```

- [ ] **Step 3: Commit**

```bash
git add test/mpi/test_slepc.jl .github/workflows/mpi.yml
git commit -m "test(mpi): verify across-k groups at n=4 (2×2) and n=2 (2×1)"
```

Verification: **CI (mpi.yml)** runs the script at n=1 (ng=1), n=2 (ng=2, 2×1), and
n=4 (ng=2, 2×2) — exercising the split, round-robin routing, per-group distributed
solve, and cross-group gather. Each `k`'s σ₁ must match `k²+π²`.

---

## Task 4: Document `ngroups`

**Files:**
- Modify: `docs/src/mpi.md`

- [ ] **Step 1: Document the kwarg and the model**

In `docs/src/mpi.md`, after the `## Usage` code block, add:

```markdown
### Parallel across wavenumbers (`ngroups`)

By default (`ngroups=1`) all ranks collaborate on one wavenumber's pencil at a
time. Set `ngroups=G` to split the `P` ranks into `G` equal groups (requires
`P % G == 0`): the wavenumbers are distributed round-robin across groups, each
group solving its subset on its own sub-communicator while the matrix of each
pencil is still distributed across that group's ranks. Results are gathered to
global rank 0.

```julia
# 8 ranks, 4 groups of 2: 4 wavenumbers solved concurrently, each pencil on 2 ranks
mpiexec -n 8 julia --project=. -e '... solve(cache, k_values; sigma_0=0.02, nev=5, ngroups=4) ...'
```

Pick `ngroups` to balance across-wavenumber concurrency (more groups) against
per-solve distribution (larger groups). Assembly of each pencil is still serial on
its group's rank 0.
```

- [ ] **Step 2: Commit**

```bash
git add docs/src/mpi.md
git commit -m "docs: document ngroups across-wavenumber parallelism"
```

---

## Task 5: Final verification

- [ ] **Step 1: Cross-platform suite green locally**

Run:
```bash
JL=/Users/subha/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia
export JULIA_DEPOT_PATH="/tmp/jdepot:/Users/subha/.julia"
"$JL" --project=. -e 'using Pkg; Pkg.test()'
```
Expected: PASS (includes the new `_group_indices` tests).

- [ ] **Step 2: ext parses**

Run: `"$JL" -e 'Meta.parseall(read("ext/BiGSTARSMPIExt.jl", String)); println("OK")'`
Expected: `OK`.

- [ ] **Step 3: Push and drive `mpi.yml` green**

Push the branch / open a PR. The blocking `mpi.yml` runs n=1/n=2/n=4. Debug any
group deadlock or numeric mismatch against the analytic reference. Iterate until
all three rank counts pass. (Commits/PR only on user authorization — standing
no-commit rule + auto-sync.)

---

## Self-review

**Spec coverage:** `ngroups` kwarg default 1 → T2; Comm_split + round-robin routing → T1 (`_group_indices`) + T2; comm-parameterize (MatCreate/EPSCreate fix) → T2 steps 1-2; point-to-point gather to global rank 0 (incl. global-0 placing its own group first) → T2 step 4; collective-safety (group_comm inside, matched send/recv gather, collective validation, empty-group skip) → T2 step 4 structure; testing (pure helper + n=4 CI) → T1, T3; docs → T4. All covered. Phase 2 (distributed assembly) correctly absent.

**Placeholder scan:** none — full code in every step, exact paths/commands.

**Type consistency:** `_group_indices(nk, ngroups, group_id)::Vector{Int}` defined in T1, called in T2/tests with matching args; `group_comm`/`grank`/`wrank`/`psize`/`group_id` consistent across T2; `_solve_one_adaptive` call signature matches the existing definition (unchanged); `MatCreate(comm)`/`EPSCreate(comm)` match the verified PetscWrap/SlepcWrap signatures; result vector is `Vector{SolverResults}` as before.
