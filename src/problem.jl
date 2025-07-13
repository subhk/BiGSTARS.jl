

struct Problem{T<:Real}
    "the problem parameters"
    params::Params{T}

    "the grid"
    grid::TwoDGrid{params.Ny, params.Nz}

    "the differential operators"
    operators::ChebMatrix{params.Ny, params.Nz}

end

function Problem{T<:Real}(params::Params{T}, 
                            grid::TwoDimGrid{params.Ny, params.Nz}, 
                            operators::ChebMatrix{params.Ny, params.Nz}) where T

    


    return Problem(params, grid, operators)
end


struct TwoDGrid{T<:AbstractFloat, Ty, Tm} <: AbstractGrid{T, Ty, Tm}
    "Number of grid points in the y-direction"  
    Ny::Int
    "Number of grid points in the z-direction"
    Nz::Int
    
    "domain extent in ``y``"
    L :: T
    "domain extent in ``z``"
    H :: T

    "range with ``y``-grid-points"
    y :: Ty
    "range with ``z``-grid-points"
    z :: Ty

    "Differentiation matrices"
    Dʸ  :: Tm
    D²ʸ :: Tm
    D⁴ʸ :: Tm

    Dᶻ  :: Tm
    D²ᶻ :: Tm
    D⁴ᶻ :: Tm
end


function TwoDGrid(Ny, L, Ny, H, T=Float64) 
    @assert Ny > 0 && Nz > 0, "Ny and Nz must be positive integers."

    # setup Fourier differentiation matrices  
    # Fourier in y-direction: y ∈ [0, L)
    y,  Dʸ  = FourierDiff(Ny, 1)
    _,  D²ʸ = FourierDiff(Ny, 2)
    _,  D⁴ʸ = FourierDiff(Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    y       = (L/2π)   * y
    Dʸ      = (2π/L)^1 * Dʸ
    D²ʸ     = (2π/L)^2 * D²ʸ
    D⁴ʸ     = (2π/L)^4 * D⁴ʸ

    # Chebyshev in the z-direction
    z,  Dᶻ  = chebdif(Nz, 1)
    _,  D²ᶻ = chebdif(Nz, 2)
    _,  D³ᶻ = chebdif(Nz, 3)
    _,  D⁴ᶻ = chebdif(Nz, 4)

    # Transform the domain and derivative operators from [-1, 1] → [0, H]
    z, Dᶻ, D²ᶻ  = chebder_transform(z,  Dᶻ, 
                                        D²ᶻ, 
                                        zerotoL_transform, 
                                        H)

    _, _, D⁴ᶻ  = chebder_transform_ho(z, Dᶻ, 
                                        D²ᶻ, 
                                        D³ᶻ, 
                                        D⁴ᶻ, 
                                        zerotoL_transform_ho, 
                                        H)

    return TwoDGrid{T, typeof(y), typeof(Dʸ)}(Ny, Nz, L, H, y, z, 
                                            Dʸ, D²ʸ, D⁴ʸ,
                                            Dᶻ, D²ᶻ, D⁴ᶻ)
end




# show(io::IO, problem::BiGSTARS.Problem) =
#     print(io, "Problem\n",
#               "  ├─────────── grid: grid (on " * string(typeof(problem.grid.device)) * ")", "\n",
#               "  ├───── parameters: params", "\n",
#               "  ├────── variables: vars", "\n",
#               "  ├─── state vector: sol", "\n",
#               "  ├─────── equation: eqn", "\n",
#               "  ├────────── clock: clock", "\n",
#               "  │                  └──── dt: ", problem.clock.dt, "\n",
#               "  └──── timestepper: ", string(nameof(typeof(problem.timestepper))))