
function ImplementBCs_Dirchilet_on_D1!(𝒟ᶻᴰ, params) 
    n = params.Nz
    𝒟ᶻᴰ[1,1]  = 0.0
    𝒟ᶻᴰ[n,n]  = 0.0
    return nothing
end

function ImplementBCs_Dirchilet_on_D2!(𝒟²ᶻᴰ, params) 
    n = params.Nz
    𝒟²ᶻᴰ[1,1] = 0.0
    𝒟²ᶻᴰ[n,n] = 0.0 
    return nothing
end

function ImplementBCs_Dirchilet_on_D4!(𝒟⁴ᶻᴰ, 𝒟²ᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        𝒟⁴ᶻᴰ[1,iter+1] = (𝒟⁴ᶻᴰ[1,iter+1] + 
                                -1.0 * 𝒟⁴ᶻᴰ[1,1] * 𝒟²ᶻ[1,iter+1])

          𝒟⁴ᶻᴰ[n,iter] = (𝒟⁴ᶻᴰ[n,iter] + 
                                -1.0 * 𝒟⁴ᶻᴰ[n,n] * 𝒟²ᶻ[n,iter])
    end
    return nothing
end

function ImplementBCs_Neumann_on_D1!(𝒟ᶻᴺ, params) 
    n = params.Nz
    @. 𝒟ᶻᴺ[1,1:end] = 0.0
    @. 𝒟ᶻᴺ[n,1:end] = 0.0
    return  nothing
end

function ImplementBCs_Neumann_on_D2!(𝒟²ᶻᴺ, 𝒟ᶻ, params) 
    n = params.Nz
    for iter ∈ 1:n-1
        𝒟²ᶻᴺ[1,iter+1] = (𝒟²ᶻᴺ[1,iter+1] + 
                                -1.0 * 𝒟²ᶻᴺ[1,1] * 𝒟ᶻ[1,iter+1]/𝒟ᶻ[1,1])

        𝒟²ᶻᴺ[n,iter]   = (𝒟²ᶻᴺ[n,iter] + 
                                -1.0 * 𝒟²ᶻᴺ[n,n] * 𝒟ᶻ[n,iter]/𝒟ᶻ[n,n])
    end

    𝒟²ᶻᴺ[1,1] = 0.0
    𝒟²ᶻᴺ[n,n] = 0.0
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

function setBCs!(diffMatrix, params, bc_type::String)
    if bc_type == "dirchilet"
        ImplementBCs_Dirchilet_on_D1!(diffMatrix.𝒟ᶻᴰ,                  params)
        ImplementBCs_Dirchilet_on_D2!(diffMatrix.𝒟²ᶻᴰ,                 params)
        ImplementBCs_Dirchilet_on_D4!(diffMatrix.𝒟⁴ᶻᴰ, diffMatrix.𝒟²ᶻ, params)
    elseif bc_type == "neumann"
        ImplementBCs_Neumann_on_D1!(diffMatrix.𝒟ᶻᴺ,                    params)
        ImplementBCs_Neumann_on_D2!(diffMatrix.𝒟²ᶻᴺ,   diffMatrix.𝒟ᶻ,  params) 
        #ImplementBCs_Neumann_on_D4!(diffMatrix, params)
    else
        error("Invalid bc type")
    end
    return nothing
end