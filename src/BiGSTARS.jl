module BiGSTARS

    # load required packages
    using Printf
    using StaticArrays
    using SparseArrays
    using SparseMatrixDicts
    using SpecialFunctions
    using Parameters
    using BenchmarkTools

    using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

    using Arpack
    using LinearMaps

    using KrylovKit

    using LinearAlgebra
    using Statistics
    using VectorInterface

    using Reexport

    # 
    export
       # Chebychev and Fourier differentiation matrices 
        ChebyshevDiffn,
        FourierDiffn,

       # Setting boundary conditions
        setBCs!,
    
       # inverse of the horizontal Laplacian
        inverse_Lap_hor,
        InverseLaplace,

        # Eigen Solver
        EigenSolver, 
        solve!, 
        get_results, 
        compare_methods!, 
        plot_convergence,
        SolverConfig, 
        SolverResults, 
        ConvergenceHistory,
        print_summary,

       # some utility functions
        print_evals,
        sort_evals,
        sort_evals_,
        remove_evals,
        remove_spurious,
        DiagM,
        field_to_spdiagm, 
        spdiag_to_field,


       # calculate gradient 
        gradient,
        derivative,

        # wrappers for vectors
        wrapvec,
        unwrapvec,
        stack,
        MinimalVec,

        # construct linear map required for eigen solver
        construct_linear_map, 

        # Construct 2D grids
        TwoDGrid, 

        # Place holder of A, B matrices for generalized eigval problem
        GEVPMatrices, 

        # compute necessary derivatives
        compute_derivatives,
        Derivatives,
        precompute!


    import Base: show, summary

    "Abstract supertype for grids."
    abstract type AbstractGrid{T, Ty, Tm} end

    "Abstract supertype for parameters."
    abstract type AbstractParams end

    # export it so tests and users can see it
    export AbstractParams

    # include("dmsuite.jl")

    include("Fourier.jl")
    include("Chebyshev.jl")

    #include("transforms.jl")
    
    # include("shift_invert_arnoldi.jl")
    # include("shift_invert_krylov.jl")
    # include("shift_invert_arpack.jl")

    include("eig_solver.jl")
    include("setBCs.jl")
    include("utils.jl")
    include("problem.jl")
    include("eig_matrices.jl")
    include("compute_derivative.jl")
    include("gradient.jl")
    include("construct_linear_map.jl")


    function Base.show(io::IO, params::AbstractParams)
        T = typeof(params.L)  # infer float type from a field
        print(io,
            "Eigen Solver Configuration \n",
            "  ├────────────────────── Float Type: $T \n",
            "  ├─────────────── Domain Size (L, H): ", (params.L, params.H), "\n",
            "  ├───────────── Resolution (Ny, Nz): ", (params.Ny, params.Nz), "\n",
            "  ├──── Boundary Conditions (w, ζ, b): ", (params.w_bc, params.ζ_bc, params.b_bc), "\n",
            "  └────────────── Eigenvalue Solver: ", params.eig_solver, "\n"
        )
    end

end # module BiGSTARS
