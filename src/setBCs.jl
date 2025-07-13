const BC_TYPE = Dict(
    (:w,  "rigid_lid")   => :dirichlet,
    (:ζ,  "free_slip")   => :neumann,
    (:ζ,  "no_slip")     => :dirichlet,
    (:b,  "zero_flux")    => :neumann,
    (:b,  "fixed")        => :dirichlet,
    (:w,  "free")        => :neumann,
    (:w,  "open")        => :neumann
)

bc_type(var::Symbol, bc::String) = get(BC_TYPE, (var, bc), error("Unknown BC: $var => \"$bc\""))


function apply_dirchilet_on_D1!(Dᶻᴰ, params) 
    n = params.Nz
    Dᶻᴰ[1,1]  = 0.0
    Dᶻᴰ[n,n]  = 0.0
    return nothing
end

function apply_dirchilet_on_D2!(D²ᶻᴰ, params) 
    n = params.Nz
    D²ᶻᴰ[1,1] = 0.0
    D²ᶻᴰ[n,n] = 0.0 
    return nothing
end

function apply_dirchilet_on_D4!(D⁴ᶻᴰ, D²ᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        D⁴ᶻᴰ[1,iter+1] = (D⁴ᶻᴰ[1,iter+1] + 
                                -1.0 * D⁴ᶻᴰ[1,1] * D²ᶻ[1,iter+1])

          D⁴ᶻᴰ[n,iter] = (D⁴ᶻᴰ[n,iter] + 
                                -1.0 * D⁴ᶻᴰ[n,n] * D²ᶻ[n,iter])
    end
    return nothing
end

function apply_neumann_on_D1!(Dᶻᴺ, params) 
    n = params.Nz
    @. Dᶻᴺ[1,1:end] = 0.0
    @. Dᶻᴺ[n,1:end] = 0.0
    return  nothing
end

function apply_neumann_on_D2!(D²ᶻᴺ, Dᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        D²ᶻᴺ[1,iter+1] = (D²ᶻᴺ[1,iter+1] + 
                                -1.0 * D²ᶻᴺ[1,1] * Dᶻ[1,iter+1]/Dᶻ[1,1])

        D²ᶻᴺ[n,iter]   = (D²ᶻᴺ[n,iter] + 
                                -1.0 * D²ᶻᴺ[n,n] * Dᶻ[n,iter]/Dᶻ[n,n])
    end

    D²ᶻᴺ[1,1] = 0.0
    D²ᶻᴺ[n,n] = 0.0
    return nothing
end


# function ImplementBCs_Neumann_on_D4!(diffMatrix, params) 
#     n = params.Nz
#     for iter ∈ 1:n-1
#         diffMatrix.𝒟⁴ᶻᴺ[1,iter+2] = (diffMatrix.𝒟⁴ᶻᴺ[1,iter+2] + 
#                                 -1.0 * diffMatrix.𝒟⁴ᶻᴺ[1,2] * diffMatrix.𝒟²ᶻᴺ[1,iter+2]/diffMatrix.𝒟²ᶻᴺ[1,2])

#         diffMatrix.𝒟⁴ᶻᴺ[n,iter]   = (diffMatrix.𝒟⁴ᶻᴺ[n+1,iter] + 
#                                 -1.0 * diffMatrix.𝒟⁴ᶻᴺ[n,n] * diffMatrix.𝒟²ᶻᴺ[n,iter]/diffMatrix.𝒟²ᶻᴺ[n,iter])
#     end

#     diffMatrix.𝒟⁴ᶻᴺ[1,1] = 0.0
#     diffMatrix.𝒟⁴ᶻᴺ[n,n] = 0.0
# end

## Set BCs Based on Type Symbol
function setBCs!(grid, params, ::Val{:dirichlet})
    n = params.Nz
    apply_dirichlet_D1!(grid.Dᶻᴰ,            n)
    apply_dirichlet_D2!(grid.D²ᶻᴰ,           n)
    apply_dirichlet_D4!(grid.D⁴ᶻᴰ, grid.D²ᶻ, n)
end

function setBCs!(grid, params, ::Val{:neumann})
    n = params.Nz
    apply_neumann_D1!(grid.Dᶻᴺ,           n)
    apply_neumann_D2!(grid.D²ᶻᴺ, grid.Dᶻ, n)
end

# function setBCs!(grid, params, bc_type::String)
#     if bc_type == "dirchilet" 
#         apply_Dirchilet_on_D1!(grid.Dᶻᴰ,             params)
#         apply_Dirchilet_on_D2!(grid.D²ᶻᴰ,            params)
#         apply_Dirchilet_on_D4!(grid.D⁴ᶻᴰ, grid.D²ᶻ,  params)

#     elseif bc_type == "neumann" 
#         apply_Neumann_on_D1!(grid.Dᶻᴺ,               params)
#         apply_Neumann_on_D2!(grid.D²ᶻᴺ, grid.Dᶻ,     params)

#     else
#         error("Unknown boundary condition type: $bc_type")
#     end
#     return nothing
# end