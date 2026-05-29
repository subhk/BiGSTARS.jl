using Test
using LinearAlgebra
using Random

@testset "Eigen solver interface" begin
    @testset "solve_eigenvalue_problem accepts verbose keyword" begin
        A = ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0]))
        B = Matrix{ComplexF64}(I, 4, 4)

        λ, Χ = BiGSTARS.solve_eigenvalue_problem(A, B;
            method=:Arpack, σ₀=2.5, verbose=false, n_tries=0, nev=1)

        @test length(λ) == 1
        @test size(Χ, 1) == 4
    end

    @testset "EigenSolver preserves concrete matrix storage types" begin
        A = ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0]))
        B = Matrix{ComplexF64}(I, 4, 4)

        solver = EigenSolver(A, B; σ₀=2.5)

        @test fieldtype(typeof(solver), :A) === typeof(A)
        @test fieldtype(typeof(solver), :B) === typeof(B)
    end

    @testset "solve! reuses caller-provided A_shifted buffer" begin
        # Diagonal problem: exact eigenvalues are 1,2,3,4,5. σ₀=2.2 → nearest is 2.0.
        A = ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0, 5.0]))
        B = Matrix{ComplexF64}(I, 5, 5)

        # Baseline: no buffer
        s1 = EigenSolver(A, B; σ₀=2.2, method=:Arnoldi, nev=1, n_tries=2)
        solve!(s1; verbose=false)
        λ1, _ = get_results(s1)

        # Buffered: caller supplies preallocated A_shifted scratch (sentinel-filled)
        buf = fill(ComplexF64(NaN), 5, 5)
        s2 = EigenSolver(A, B; σ₀=2.2, method=:Arnoldi, nev=1, n_tries=2)
        solve!(s2; verbose=false, A_buf=buf)
        λ2, _ = get_results(s2)

        truevals = ComplexF64.(1:5)
        @test !any(isnan, buf)                              # buffer was actually used (overwritten)
        @test minimum(abs.(λ1[1] .- truevals)) < 1e-6       # baseline finds a true eigenvalue
        @test minimum(abs.(λ2[1] .- truevals)) < 1e-6       # buffered finds a true eigenvalue
        @test abs(λ1[1] - λ2[1]) < 1e-10                    # buffer path identical to baseline

        # Mismatched buffer size is rejected
        @test_throws DimensionMismatch solve!(
            EigenSolver(A, B; σ₀=2.2, method=:Arnoldi, nev=1, n_tries=2);
            verbose=false, A_buf=zeros(ComplexF64, 4, 4))
    end

    @testset "Krylov caps krylovdim at problem size" begin
        N = 40
        Random.seed!(1); M = rand(ComplexF64, N, N)
        A = M + M'; B = Matrix{ComplexF64}(I, N, N)
        runk(kd) = (Random.seed!(3);
                    s = EigenSolver(A, B; σ₀=0.5, method=:Krylov, nev=1, n_tries=1, krylovdim=kd);
                    solve!(s; verbose=false);
                    s.results.eigenvalues[1])

        runk(N); runk(200)                          # warmup (compile)
        Random.seed!(3); aN   = @allocated runk(N)
        Random.seed!(3); a200 = @allocated runk(200)

        # krylovdim > N must be clamped to N: no oversized Krylov basis allocation
        @test a200 ≤ 1.5 * aN
        # and the clamp does not change the answer
        @test abs(runk(N) - runk(200)) < 1e-8
    end

    @testset "krylovdim default is modest and overridable" begin
        @test SolverConfig(σ₀=1.0).krylovdim == 30          # sane default (was 200)
        @test SolverConfig(σ₀=1.0, krylovdim=150).krylovdim == 150  # explicit value kept

        # The runtime clamp guarantees krylovdim > nev (KrylovKit requirement):
        # a user-supplied krylovdim=2 with nev=3 is bumped to nev+2=5, so the
        # solve runs instead of throwing.
        A = ComplexF64.(Diagonal(collect(1.0:12.0))); B = Matrix{ComplexF64}(I, 12, 12)
        s = EigenSolver(A, B; σ₀=6.0, method=:Krylov, nev=3, n_tries=1, krylovdim=2)
        solve!(s; verbose=false)
        @test s.results.converged
    end

    @testset "sortby=:nearest returns the mode at the shift" begin
        @test SolverConfig(σ₀=1.0).sortby == :nearest          # default

        # diag spectrum 1..5, shift 2.2 → nearest true eigenvalue is 2.0
        A = ComplexF64.(Diagonal([1.0, 2.0, 3.0, 4.0, 5.0])); B = Matrix{ComplexF64}(I, 5, 5)
        s = EigenSolver(A, B; σ₀=2.2, method=:Arnoldi, nev=3, n_tries=0)  # default :nearest
        solve!(s; verbose=false)
        λ, _ = get_results(s)
        @test abs(λ[1] - 2.0) < 1e-6                            # nearest to σ₀ reported first

        # :nearest without σ is an error
        @test_throws ArgumentError BiGSTARS.sort_eigenvalues!(
            ComplexF64[1.0, 2.0], Matrix{ComplexF64}(I, 2, 2), :nearest)
    end
end
