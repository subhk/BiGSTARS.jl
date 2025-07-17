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


function apply_dirichlet_on_D1!(Dᶻᴰ, params) 
    n = params.Nz
    Dᶻᴰ[1,1]  = 0.0
    Dᶻᴰ[n,n]  = 0.0
    return nothing
end

function apply_dirichlet_on_D2!(D²ᶻᴰ, params) 
    n = params.Nz
    D²ᶻᴰ[1,1] = 0.0
    D²ᶻᴰ[n,n] = 0.0 
    return nothing
end

function apply_dirichlet_on_D3!(D³ᶻᴰ, params) 
    n = params.Nz
    D³ᶻᴰ[1,1] = 0.0
    D³ᶻᴰ[n,n] = 0.0 
    return nothing
end

function apply_dirichlet_on_D4!(D⁴ᶻᴰ, D²ᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        D⁴ᶻᴰ[1,iter+1] = (D⁴ᶻᴰ[1,iter+1] + 
                                -1.0 * D⁴ᶻᴰ[1,1] * D²ᶻ[1,iter+1])

          D⁴ᶻᴰ[n,iter] = (D⁴ᶻᴰ[n,iter] + 
                                -1.0 * D⁴ᶻᴰ[n,n] * D²ᶻ[n,iter])
    end

    D⁴ᶻᴰ[1,1] = 0.0
    D⁴ᶻᴰ[n,n] = 0.0 
    
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

function apply_neumann_on_D3!(D³ᶻᴺ, Dᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        D³ᶻᴺ[1,iter+1] = (D³ᶻᴺ[1,iter+1] + 
                            -1.0 * D³ᶻᴺ[1,1] * Dᶻ[1,iter+1]/Dᶻ[1,1])

        D³ᶻᴺ[n,iter]   = (D³ᶻᴺ[n,iter] + 
                            -1.0 * D³ᶻᴺ[n,n] * Dᶻ[n,iter]/Dᶻ[n,n])
    end

    D³ᶻᴺ[1,1] = 0.0
    D³ᶻᴺ[n,n] = 0.0
    return nothing
end


function apply_neumann_on_D4!(D⁴ᶻᴺ, Dᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        D⁴ᶻᴺ[1,iter+1] = (D³ᶻᴺ[1,iter+1] + 
                            -1.0 * D³ᶻᴺ[1,1] * Dᶻ[1,iter+1]/Dᶻ[1,1])

        D⁴ᶻᴺ[n,iter]   = (D³ᶻᴺ[n,iter] + 
                            -1.0 * D³ᶻᴺ[n,n] * Dᶻ[n,iter]/Dᶻ[n,n])
    end

    D⁴ᶻᴺ[1,2] = 0.0
    D⁴ᶻᴺ[n,n] = 0.0
    return nothing
end


function setBCs!(grid, params, bc::Symbol)
    if bc == :dirichlet
        apply_dirichlet_on_D1!(grid.Dᶻᴰ,            params)
        apply_dirichlet_on_D2!(grid.D²ᶻᴰ,           params)
        apply_dirichlet_on_D4!(grid.D⁴ᶻᴰ, grid.D²ᶻ, params)
    elseif bc == :neumann
        apply_neumann_on_D1!(grid.Dᶻᴺ,              params)
        apply_neumann_on_D2!(grid.D²ᶻᴺ,   grid.Dᶻ,  params)
    else
        error("Unknown BC symbol: $bc")
    end
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