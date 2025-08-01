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
       # Chebychev matrix
    #    cheb,
    #    chebdif,   
    #    chebder_transform,
    #    chebder_transform_ho,  
    #    MinusLtoPlusL_transform, 
    #    MinusLtoZero_transform,
    #    zerotoL_transform, 
    #    transform_02π_to_0L,
    #    zerotoL_transform_ho, 
    #    zerotoone_transform, 
    #    cheb_coord_transform, 
    #    cheb_coord_transform_ho,   

        ChebyshevDiffn,

       # Fourier differentiation matrix
    #    FourierDiff,
    #    FourierDiff_fdm, 

        FourierDiffn,

       # Setting boundary conditions
        setBCs!,
    
       # inverse of the horizontal Laplacian
        inverse_Lap_hor,

       # eigenvalue solvers
    #    Eigs_Arnoldi, 
    #    Eigs_Krylov,
    #    Eigs_Arpack,
    #    solve_shift_invert_arnoldi,
    #    solve_shift_invert_krylov,
    #    solve_shift_invert_arpack,

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

       # calculate gradient 
        gradient,
        derivative,

        # wrappers for vectors
        wrapvec,
        unwrapvec,
        stack,
        MinimalVec,

        # pardiso_solver to construct linear map
        #construct_linear_map_pardiso,
        construct_linear_map, 

        # Construct 2D grids
        TwoDGrid, 

        # Place holder of A, B matrices for generalized eigval problem
        GEVPMatrices, 

        # compute necessary derivatives
        compute_derivatives,
        compute_derivatives_legacy,  
        Derivatives,
        precompute!

        # # initialize the basic state
        # initialize_basic_state_from_fields,
        # initialize_basic_state!


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
