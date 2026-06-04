# Run with: mpiexec -n {1,2} julia --project=test/mpi test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (PETSC_DIR/PETSC_ARCH/SLEPC_DIR)
# and MPI.jl bound to the same MPI.
#
# `solve` manages SlepcInitialize itself (the static solver options go into the
# PETSc options database at init time), so this script does not call it.
# PetscWrap/SlepcWrap are imported only to ACTIVATE the BiGSTARSMPIExt extension
# (which provides the real `solve`). Use `import`, NOT `using`: PetscWrap exports
# `solve`, which would shadow BiGSTARS.solve used below.
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap
using Test
using LinearAlgebra

# MPI.jl's COMM_WORLD handle is populated only by its own MPI.Init(); calling
# MPI.Comm_rank before it aborts ("MPI_Comm_rank called before MPI_INIT").
# `solve` also guards init, but this script touches the communicator first.
MPI.Initialized() || MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)

# ── Analytic reference (no serial solver exists anymore) ──────────────────────
# σ u = -dx²u - dz²u, u(0)=u(1)=0 on z∈[0,1], at wavenumber k.
# dx → ×k² (FourierTransformed); -dz² has eigenvalues (nπ)² ⇒ σ_n = k² + (nπ)².
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=24, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

k = 1.0
analytic(n) = k^2 + (n * π)^2          # σ_1 ≈ 10.8696, σ_2 ≈ 40.48, σ_3 ≈ 89.83

# SLEPc options (nev/which/tol/…) enter the PETSc database once per process and
# CANNOT change between solves; only the σ target may (via EPSSetTarget). So every
# solve() in this script MUST use the SAME nev/which/tol. One nev=4 solve covers
# both the σ_1 and σ_1..3 checks.
res = solve(cache, [k]; sigma_0=10.0, nev=4, which=:LM, tol=1e-10)

if rank == 0
    A, B = BiGSTARS.assemble(cache, k)   # serial matrices for the residual check
    ts = @testset "SLEPc analytic reference" begin
        @test res[1].converged
        λ1 = res[1].eigenvalues[1]
        @test isapprox(real(λ1), analytic(1); rtol=1e-4)   # nearest the shift = σ_1
        @test abs(imag(λ1)) < 1e-6

        χ = res[1].eigenvectors[:, 1]
        @test norm(A * χ - λ1 * (B * χ)) / norm(χ) < 1e-6  # residual (also checks the gather)

        @test length(res[1].eigenvalues) ≥ 3
        got = sort(real.(res[1].eigenvalues))
        for n in 1:3
            @test minimum(abs.(got .- analytic(n))) < 1e-3  # σ_1,σ_2,σ_3 all present
        end

        @test !isempty(res[1].history.attempts)            # adaptive history populated
        println("σ_1 SLEPc=$(real(λ1))  analytic=$(analytic(1))")
    end
    ts.anynonpass && exit(1)
end

# ── Spurious-mode filter on a singular-B descriptor system ────────────────────
# Augmented derived system has zero rows in B (constraint + BC rows) → infinite
# modes; the filter must drop them. Physical σ = -1/(nπ)²; nearest -0.1 is -1/π².
dom2 = Domain(z = Chebyshev(N=48, lower=0.0, upper=1.0))
prob2 = EVP(dom2, variables=[:psi], eigenvalue=:sigma)
@derive prob2 v dz(dz(v)) = psi
@derive_bc prob2 v left(v) == 0
@derive_bc prob2 v right(v) == 0
@equation prob2 sigma * psi == v
@bc prob2 left(psi) == 0
@bc prob2 right(psi) == 0
cache2 = discretize(prob2; augment_derived=true)

# Same static options (nev/which/tol) as the solve above — required, since they
# are fixed once per process. Only sigma_0 differs (set via EPSSetTarget).
res_sp = solve(cache2; sigma_0=-0.1, nev=4, which=:LM, tol=1e-10, n_tries=2)
if rank == 0
    ts2 = @testset "SLEPc spurious-mode filter (singular B)" begin
        @test res_sp[1].converged
        @test all(e -> abs(e) < 0.5, res_sp[1].eigenvalues)              # no infinite modes survive
        @test minimum(abs.(res_sp[1].eigenvalues .- (-1 / π^2))) < 1e-3  # physical n=1 present
    end
    ts2.anynonpass && exit(1)
end

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
