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
       cheb,
       chebdif,   
       chebder_transform,
       chebder_transform_ho,  
       MinusLtoPlusL_transform, 
       MinusLtoZero_transform,
       zerotoL_transform, 
       transform_02π_to_0L,
       zerotoL_transform_ho, 
       zerotoone_transform, 
       cheb_coord_transform, 
       cheb_coord_transform_ho,   

       # Fourier differentiation matrix
       FourierDiff,
       FourierDiff_fdm,

       # Constructing derivative operators
       #Construct_DerivativeOperator!,
       #ImplementBCs_cheb!,  

       # Setting boundary conditions
       setBCs!,
    
       # inverse of the horizontal Laplacian
       inverse_Lap_hor,

       # eigenvalue solvers
       Eigs_Arnoldi, 
       Eigs_Krylov,
       Eigs_Arpack,
       solve_shift_invert_arnoldi,
       solve_shift_invert_krylov,
       solve_shift_invert_arpack,

       # some utility functions
       print_evals,
       sort_evals,
       remove_evals,
       remove_spurious,
       gradient,
       gradient2, 

       # wrappers for vectors
       wrapvec,
       unwrapvec,
       stack,
       MinimalVec,

       # pardiso_solver to construct linear map
       #construct_linear_map_pardiso,
       construct_linear_map, 

       TwoDGrid, 

       GEVPMatrices


    import Base: show, summary

    "Abstract supertype for grids."
    abstract type AbstractGrid{T, Ty, Tm} end

    "Abstract supertype for parameters."
    abstract type AbstractParams end

    include("dmsuite.jl")
    include("transforms.jl")
    include("shift_invert_arnoldi.jl")
    include("shift_invert_krylov.jl")
    include("shift_invert_arpack.jl")
    include("setBCs.jl")
    include("utils.jl")
    include("problem.jl")
    include("matrices.jl")


end # module BiGSTARS
