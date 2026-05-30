module BiGSTARS

    using Printf
    using SparseArrays
    using LinearAlgebra
    using FFTW
    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM
    using Arpack
    using LinearMaps
    using KrylovKit
    using VectorInterface: MinimalSVec, MinimalMVec, MinimalVec
    using PrecompileTools: @setup_workload, @compile_workload

    export
        # Domain and coordinate types
        Domain,
        FourierTransformed,
        Fourier,
        Chebyshev,
        gridpoints,
        meshgrid,

        # Problem types
        EVP,

        # DSL macros
        @equation,
        @bc,
        @substitution,
        @derive,
        @derive_bc,

        # Discretization and solving
        discretize,
        assemble,
        assemble!,
        allocate_workspace,
        AssemblyWorkspace,
        solve,
        DiscretizationCache,
        reconstruct,
        reconstruct_all,
        evaluate_field,
        @compute,
        @compute_setup,

        # Eigenvalue solver
        EigenSolver,
        solve!,
        get_results,
        SolverConfig,
        SolverResults,
        ConvergenceHistory,
        compare_methods!,
        solve_mpi,
        print_summary,

        # Utilities
        print_evals,
        sort_evals,
        remove_evals,
        differentiate,
        to_coefficients,
        to_physical,
        chebyshev_points,
        chebyshev_coefficients,
        chebyshev_evaluate,
        fourier_points

    # Core spectral operators (coefficient-space)
    include("ultraspherical.jl")
    include("fourier_coeff.jl")

    # Domain and problem types
    include("domain.jl")

    # Transforms (after domain.jl — differentiate needs Domain types)
    include("transforms.jl")
    include("expr.jl")
    include("evp.jl")

    # DSL
    include("macros.jl")
    include("substitutions.jl")
    include("lowering.jl")
    include("k_separation.jl")
    include("boundary.jl")

    # Eigenvalue solver
    include("construct_linear_map.jl")
    include("eig_solver.jl")
    include("mpi_prep.jl")

    # Discretization and solving
    include("discretize.jl")
    include("solve.jl")
    include("reconstruct.jl")

    # Utilities
    include("utils.jl")

    # ──────────────────────────────────────────────────────────────────────────
    #  Precompilation: exercise discretize → assemble → solve on a tiny problem
    #  so the first real call doesn't pay full compile latency (TTFX).
    # ──────────────────────────────────────────────────────────────────────────
    @setup_workload begin
        @compile_workload begin
            dom = Domain(x = FourierTransformed(), z = Chebyshev(N=8, lower=0.0, upper=1.0))
            prob = EVP(dom, variables=[:u], eigenvalue=:sigma)
            @equation prob sigma * u == -dx(dx(u)) - dz(dz(u))
            @bc prob left(u) == 0
            @bc prob right(u) == 0
            cache = discretize(prob)
            # `solve` swallows non-convergence (returns a failed result), so the
            # workload never throws during precompilation regardless of the shift.
            solve(cache, [1.0]; sigma_0 = 10.0, method = :Krylov, nev = 1, verbose = false)
        end
    end

end # module BiGSTARS
