# Run with: mpiexec -n 2 julia --project=. test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (SLEPC_DIR/PETSC_DIR/PETSC_ARCH).
using BiGSTARS
using MPI, PetscWrap, SlepcWrap
using Test
using LinearAlgebra

SlepcInitialize()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Small Poisson-type EVP with a known serial answer.
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=16, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

# Distributed solve.
res = solve_mpi(cache, [1.0]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)

# Serial reference on rank 0.
if rank == 0
    ref = solve(cache, [1.0]; sigma_0=10.0, method=:Krylov, nev=1, verbose=false)
    @test res[1].converged
    @test ref[1].converged
    λ_mpi = res[1].eigenvalues[1]
    λ_ser = ref[1].eigenvalues[1]
    @test abs(λ_mpi - λ_ser) < 1e-6
    println("MPI λ=$(λ_mpi)  serial λ=$(λ_ser)  ✓")
end

SlepcFinalize()
