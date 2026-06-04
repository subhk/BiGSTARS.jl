# eady_mpi.jl — distributed (MPI) Eady baroclinic instability via `solve`.
#
# `solve` is the package's eigensolver (SLEPc over PETSc): the eigenproblem is
# spread across MPI ranks. Same problem as `examples/Eady.jl`.
#
# ── Prerequisites ────────────────────────────────────────────────────────────
# 1. A complex-scalar system PETSc + SLEPc build, with PETSC_DIR / PETSC_ARCH /
#    SLEPC_DIR exported:
#        ./configure --with-scalar-type=complex --download-mumps --download-scalapack
# 2. MPI.jl bound to the SAME MPI used to build PETSc, in an environment that has
#    BiGSTARS + MPI + PetscWrap + SlepcWrap (the repo's `test/mpi` env works):
#        julia --project=test/mpi -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#        julia --project=test/mpi -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
#
# ── Run ──────────────────────────────────────────────────────────────────────
#        mpiexec -n 4 julia --project=test/mpi examples/eady_mpi.jl
#
# `solve` initializes SLEPc itself (the static solver options must be in the PETSc
# options database at init time), so you do NOT call SlepcInitialize/SlepcFinalize.

using BiGSTARS
using MPI
import PetscWrap, SlepcWrap   # import (not using): PetscWrap exports `solve`, which would shadow BiGSTARS.solve
using Printf

# ── Build the EVP (runs on every rank; only rank-0's matrices get used) ──
domain = Domain(
    x = FourierTransformed(),     # along-front → wavenumber k
    y = Fourier(60, [0, 1]),      # cross-front, periodic
    z = Chebyshev(40, [0, 1]),    # vertical, rigid lids
)

prob = EVP(domain, variables=[:psi], eigenvalue=:sigma)

Y, Z = gridpoints(domain, :y, :z)
Ri = 1.0
prob[:U]    = Z .- 0.5            # along-front velocity U(z) = z - 1/2
prob[:Ri]   = Ri                  # Richardson number (N² = Ri)
prob[:E]    = 1e-12               # hyperviscosity
prob[:dBdy] = -ones(length(Z))   # dB/dy = -1 for the Eady basic state
prob[:dQdy] = zeros(length(Z))   # interior PV gradient (zero for Eady)

@substitution Lap(A) = dx(dx(A)) + dy(dy(A)) + Ri * dz(dz(A))
@equation sigma * Lap(psi) = U * dx(Lap(psi)) + dQdy * dx(psi) - E * Lap(Lap(psi))
@bc left(sigma * dz(psi)  + U * dx(dz(psi)) + dBdy * dx(psi)) = 0
@bc right(sigma * dz(psi) + U * dx(dz(psi)) + dBdy * dx(psi)) = 0

cache = discretize(prob)

# ── Distributed solve: one eigenproblem at k = 1.0, across all ranks ──
# `sigma_0` is the shift-and-invert target (a guess near the growth rate);
# the `nev` modes nearest it come back, and rank 0 picks the most unstable.
# solve manages SlepcInitialize (with the solver options) on first call.
results = solve(cache, [1.0];
                    sigma_0    = 0.2,     # target shift
                    nev        = 6,
                    which      = :LM,     # eigenvalues nearest the shift
                    tol        = 1e-10,
                    mat_solver = :mumps)  # parallel direct solver for the inner solves

# ── Only rank 0 has populated results; other ranks get empty markers ──
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    r = results[1]
    if r.converged
        i = argmax(real.(r.eigenvalues))   # most unstable = largest growth rate
        σ = r.eigenvalues[i]
        @printf("k=1.00  most unstable σ = %.6f %+.6fi  (growth %.6f)\n",
                real(σ), imag(σ), real(σ))
    else
        println("did not converge")
    end
end
