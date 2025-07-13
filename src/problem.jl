

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


struct TwoDGrid{Ny, Nz, params} 
    "Number of grid points in the y-direction"  
    Ny::Int
    "Number of grid points in the z-direction"
    Nz::Int
    "The problem parameters"
    params::Params
end


function TwoDGrid(Ny, Nz, params) 
    # Ensure Ny and Nz are positive integers
    if Ny <= 0 || Nz <= 0
        throw(ArgumentError("Ny and Nz must be positive integers."))
    end

    y = LinRange(0, 1, Ny)  # y-coordinates
    z = LinRange(0, 1, Nz)  # z-coordinates

    return TwoDGrid(Ny, Nz)
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