# Run with: mpiexec -n {1,2,4} julia --project=test/mpi test/mpi/test_slepc_reuse.jl
# Validates the reuse_factorization=true solve path (persistent A/B/eps across a group's
# wavenumbers, MUMPS ordering reused) against the analytic reference.
#
# reuse_factorization changes the PETSc options string, which is set ONCE per process — so
# this runs in its OWN process (separate from test_slepc.jl, which exercises the non-reuse
# path) and checks reuse against analytic σ_n directly, rather than diffing two solves.
using BiGSTARS
using MPI
import PetscWrap, SlepcWrap
using Test
using LinearAlgebra

MPI.Initialized() || MPI.Init()
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

# σ u = -dx²u - dz²u, u(0)=u(1)=0 ⇒ σ_1(k) = k² + π² (Poisson pencil).
dom = Domain(x = FourierTransformed(), z = Chebyshev(N=24, lower=0.0, upper=1.0))
prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
@equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
@bc prob left(u) == 0
@bc prob right(u) == 0
cache = discretize(prob)

ng = (nprocs % 2 == 0) ? 2 : 1
ks = [0.5, 1.0, 1.5, 2.0]

# Reuse path on a full cache: A/B/eps held across the group's k-subset.
res = solve(cache, ks; sigma_0=10.0, nev=4, which=:LM, tol=1e-10,
            ngroups=ng, reuse_factorization=true)
if rank == 0
    ts = @testset "reuse_factorization analytic (ngroups=$(ng))" begin
        @test length(res) == length(ks)
        for (j, kj) in enumerate(ks)
            @test res[j].converged
            σ1 = minimum(real, res[j].eigenvalues)            # smallest = k² + π²
            @test isapprox(σ1, kj^2 + π^2; rtol=1e-3)
        end
        println("reuse ng=$(ng): " *
                join(["k=$(ks[j]) σ1=$(minimum(real,res[j].eigenvalues))" for j in 1:length(ks)], "  "))
    end
    ts.anynonpass && exit(1)
end

# Reuse path on a per-rank row-restricted (distributed) cache.
dcache = discretize_distributed(prob; ngroups=ng)
res_d = solve(dcache, ks; sigma_0=10.0, nev=4, which=:LM, tol=1e-10,
              ngroups=ng, reuse_factorization=true)
if rank == 0
    ts2 = @testset "reuse + distributed cache (ngroups=$(ng))" begin
        @test length(res_d) == length(ks)
        for (j, kj) in enumerate(ks)
            @test res_d[j].converged
            @test isapprox(minimum(real, res_d[j].eigenvalues), kj^2 + π^2; rtol=1e-3)
        end
    end
    ts2.anynonpass && exit(1)
end

# Guard: reuse_factorization=true must REJECT a cache with derived-variable terms
# (augment_derived=false ⇒ non-empty derived_caches ⇒ dense H(k) block not in the union
# preallocation). The guard runs on every rank before any collective, so a plain try/catch
# on all ranks is deadlock-safe.
dom2 = Domain(z = Chebyshev(N=48, lower=0.0, upper=1.0))
prob2 = EVP(dom2, variables=[:psi], eigenvalue=:sigma)
@derive prob2 v dz(dz(v)) = psi
@derive_bc prob2 v left(v) == 0
@derive_bc prob2 v right(v) == 0
@equation prob2 sigma * psi == v
@bc prob2 left(psi) == 0
@bc prob2 right(psi) == 0
cache2 = discretize(prob2; augment_derived=false)
threw = false
try
    solve(cache2, [1.0]; sigma_0=-0.1, nev=4, which=:LM, tol=1e-10, reuse_factorization=true)
catch
    global threw = true
end
if rank == 0
    ts3 = @testset "reuse guard rejects derived cache" begin
        @test threw
    end
    ts3.anynonpass && exit(1)
end
