using Test
using BiGSTARS
using SparseArrays
using LinearAlgebra
using FFTW

@testset "BiGSTARS.jl" begin
    include("test_ultraspherical.jl")
    include("test_fourier_coeff.jl")
    include("test_eig_solver.jl")
    include("test_transforms.jl")
    include("test_domain.jl")
    include("test_expr.jl")
    include("test_evp.jl")
    include("test_macros.jl")
    include("test_substitutions.jl")
    include("test_lowering.jl")
    include("test_k_separation.jl")
    include("test_boundary.jl")
    include("test_discretize.jl")
    include("test_integration.jl")
end
