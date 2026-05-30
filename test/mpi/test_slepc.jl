# Run with: mpiexec -n {1,2} julia --project=test/mpi test/mpi/test_slepc.jl
# Requires a complex-scalar system PETSc/SLEPc (PETSC_DIR/PETSC_ARCH/SLEPC_DIR)
# and MPI.jl bound to the same MPI via MPIPreferences.use_system_binary().
#
# solve_mpi manages SlepcInitialize itself (the solver options go into the PETSc
# options database at init time), so this script does not call SlepcInitialize.
using BiGSTARS
using MPI
# PetscWrap & SlepcWrap are loaded only to activate BiGSTARSMPIExt. Use `import`,
# not `using`: PetscWrap also exports `solve`, which would shadow BiGSTARS.solve
# (used below) and error with an ambiguous-binding UndefVarError.
import PetscWrap, SlepcWrap
using Test
using LinearAlgebra

# Small Poisson-type EVP with a known serial answer (built on every rank).
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=16, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

# Distributed solve (collective across all ranks); solve_mpi initializes SLEPc.
res = solve_mpi(cache, [1.0]; sigma_0=10.0, nev=1, which=:LM, tol=1e-10)

# Verify on rank 0 against the serial :Krylov result and the eigenpair residual.
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    ref = solve(cache, [1.0]; sigma_0=10.0, method=:Krylov, nev=1, verbose=false)
    A, B = BiGSTARS.assemble(cache, 1.0)        # serial matrices for the residual

    ts = @testset "solve_mpi vs serial" begin
        @test res[1].converged
        @test ref[1].converged

        λ_mpi = res[1].eigenvalues[1]
        λ_ser = ref[1].eigenvalues[1]
        @test abs(λ_mpi - λ_ser) < 1e-6         # eigenvalue matches serial

        χ = res[1].eigenvectors[:, 1]           # gathered eigenvector
        resid = norm(A * χ - λ_mpi * (B * χ)) / norm(χ)
        @test resid < 1e-6                      # phase-independent: checks the gather too

        println("MPI λ=$(λ_mpi)  serial λ=$(λ_ser)  residual=$(resid)")
    end

    # Make CI go red on any failure (a bare @test does not set the exit code).
    ts.anynonpass && exit(1)
end
