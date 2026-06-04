module BiGSTARS

    using Printf
    using SparseArrays
    using LinearAlgebra
    using FFTW
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

        # Eigenvalue solver (SLEPc/PETSc backend; `solve` is exported above)
        SolverResults,
        ConvergenceHistory,
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

    # Eigenproblem result types (shared by the solver and reconstruction)
    include("results.jl")

    # Pure-Julia matrix prep for the distributed backend
    include("mpi_prep.jl")

    # Discretization (defines DiscretizationCache + assemble)
    include("discretize.jl")

    # Public solve entrypoint (stub + fallback; real impl in ext/BiGSTARSMPIExt.jl)
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
            assemble(cache, 1.0)   # exercise discretize→assemble; no eigensolve (needs SLEPc)
        end
    end

end # module BiGSTARS
