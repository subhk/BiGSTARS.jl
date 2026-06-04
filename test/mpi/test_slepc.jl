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

res1 = solve(cache, [k]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)
res3 = solve(cache, [k]; sigma_0=10.0, nev=3, which=:LM, tol=1e-10)

if rank == 0
    A, B = BiGSTARS.assemble(cache, k)   # serial matrices for the residual check
    ts = @testset "SLEPc analytic reference" begin
        @test res1[1].converged
        λ1 = res1[1].eigenvalues[1]
        @test isapprox(real(λ1), analytic(1); rtol=1e-4)   # matches analytic σ_1
        @test abs(imag(λ1)) < 1e-6

        χ = res1[1].eigenvectors[:, 1]
        @test norm(A * χ - λ1 * (B * χ)) / norm(χ) < 1e-6  # residual (also checks the gather)

        @test res3[1].converged
        @test length(res3[1].eigenvalues) ≥ 3
        got = sort(real.(res3[1].eigenvalues[1:3]))
        for n in 1:3
            @test minimum(abs.(got .- analytic(n))) < 1e-3  # σ_1,σ_2,σ_3 all present
        end

        @test !isempty(res1[1].history.attempts)            # adaptive history populated
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

res_sp = solve(cache2; sigma_0=-0.1, nev=4, n_tries=2)
if rank == 0
    ts2 = @testset "SLEPc spurious-mode filter (singular B)" begin
        @test res_sp[1].converged
        @test all(e -> abs(e) < 0.5, res_sp[1].eigenvalues)              # no infinite modes survive
        @test minimum(abs.(res_sp[1].eigenvalues .- (-1 / π^2))) < 1e-3  # physical n=1 present
    end
    ts2.anynonpass && exit(1)
end
