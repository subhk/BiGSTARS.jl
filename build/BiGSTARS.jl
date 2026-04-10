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

    export
        # Domain and coordinate types
        Domain,
        FourierTransformed,
        Fourier,
        Chebyshev,
        gridpoints,

        # Problem types
        EVP,

        # DSL macros
        @equation,
        @bc,
        @substitution,

        # Discretization and solving
        discretize,
        assemble,
        assemble!,
        allocate_workspace,
        AssemblyWorkspace,
        solve,
        DiscretizationCache,

        # Eigenvalue solver
        EigenSolver,
        solve!,
        get_results,
        SolverConfig,
        SolverResults,
        ConvergenceHistory,
        compare_methods!,
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

    # Discretization and solving
    include("discretize.jl")
    include("solve.jl")

    # Utilities
    include("utils.jl")

end # module BiGSTARS
