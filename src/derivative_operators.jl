

# Construct the derivative operator
function Construct_DerivativeOperator!(diffMatrix, grid, params)
    N = params.Ny * params.Nz

    # ------------- setup differentiation matrices  -------------------
    # Fourier in y-direction: y ∈ [0, L)
    y1, diffMatrix.𝒟ʸ  = FourierDiff(params.Ny, 1)
    _,  diffMatrix.𝒟²ʸ = FourierDiff(params.Ny, 2)
    _,  diffMatrix.𝒟⁴ʸ = FourierDiff(params.Ny, 4)

    # Transform the domain and derivative operators from [0, 2π) → [0, L)
    grid.y         = params.L/2π  * y1
    diffMatrix.𝒟ʸ  = (2π/params.L)^1 * diffMatrix.𝒟ʸ
    diffMatrix.𝒟²ʸ = (2π/params.L)^2 * diffMatrix.𝒟²ʸ
    diffMatrix.𝒟⁴ʸ = (2π/params.L)^4 * diffMatrix.𝒟⁴ʸ

    @assert maximum(grid.y) ≈ params.L && minimum(grid.y) ≈ 0.0

    # Chebyshev in the z-direction
    # z, diffMatrix.𝒟ᶻ  = cheb(params.Nz-1)
    # grid.z = z
    # diffMatrix.𝒟²ᶻ = diffMatrix.𝒟ᶻ  * diffMatrix.𝒟ᶻ
    # diffMatrix.𝒟⁴ᶻ = diffMatrix.𝒟²ᶻ * diffMatrix.𝒟²ᶻ

    z1, D1z = chebdif(params.Nz, 1)
    _,  D2z = chebdif(params.Nz, 2)
    _,  D3z = chebdif(params.Nz, 3)
    _,  D4z = chebdif(params.Nz, 4)
    # Transform the domain and derivative operators from [-1, 1] → [0, H]
    grid.z, diffMatrix.𝒟ᶻ, diffMatrix.𝒟²ᶻ  = chebder_transform(z1,  D1z, 
                                                                    D2z, 
                                                                    zerotoL_transform, 
                                                                    params.H)
    _, _, diffMatrix.𝒟⁴ᶻ = chebder_transform_ho(z1, D1z, 
                                                    D2z, 
                                                    D3z, 
                                                    D4z, 
                                                    zerotoL_transform_ho, 
                                                    params.H)
    
    #@printf "size of Chebyshev matrix: %d × %d \n" size(diffMatrix.𝒟ᶻ)[1]  size(diffMatrix.𝒟ᶻ)[2]

    @assert maximum(grid.z) ≈ params.H && minimum(grid.z) ≈ 0.0

    return nothing
end


function ImplementBCs_cheb!(Op, diffMatrix, params)
    Iʸ = sparse(Matrix(1.0I, params.Ny, params.Ny)) 
    Iᶻ = sparse(Matrix(1.0I, params.Nz, params.Nz)) 

    # Cheb matrix with Dirichilet boundary condition
    @. diffMatrix.𝒟ᶻᴰ  = diffMatrix.𝒟ᶻ 
    @. diffMatrix.𝒟²ᶻᴰ = diffMatrix.𝒟²ᶻ 
    @. diffMatrix.𝒟⁴ᶻᴰ = diffMatrix.𝒟⁴ᶻ 

    # Cheb matrix with Neumann boundary condition
    @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ  
    @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ 

    # n = params.Nz
    # for iter ∈ 1:n-1
    #     diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] = (diffMatrix.𝒟⁴ᶻᴰ[1,iter+1] + 
    #                             -1.0 * diffMatrix.𝒟⁴ᶻᴰ[1,1] * diffMatrix.𝒟²ᶻᴰ[1,iter+1])

    #       diffMatrix.𝒟⁴ᶻᴰ[n,iter] = (diffMatrix.𝒟⁴ᶻᴰ[n,iter] + 
    #                             -1.0 * diffMatrix.𝒟⁴ᶻᴰ[n,n] * diffMatrix.𝒟²ᶻᴰ[n,iter])
    # end

    # diffMatrix.𝒟ᶻᴰ[1,1]  = 0.0
    # diffMatrix.𝒟ᶻᴰ[n,n]  = 0.0

    # diffMatrix.𝒟²ᶻᴰ[1,1] = 0.0
    # diffMatrix.𝒟²ᶻᴰ[n,n] = 0.0   

    # diffMatrix.𝒟⁴ᶻᴰ[1,1] = 0.0
    # diffMatrix.𝒟⁴ᶻᴰ[n,n] = 0.0  

    # # Neumann boundary condition
    # @. diffMatrix.𝒟ᶻᴺ  = diffMatrix.𝒟ᶻ 
    # @. diffMatrix.𝒟²ᶻᴺ = diffMatrix.𝒟²ᶻ
    # for iter ∈ 1:n-1
    #     diffMatrix.𝒟²ᶻᴺ[1,iter+1] = (diffMatrix.𝒟²ᶻᴺ[1,iter+1] + 
    #                             -1.0 * diffMatrix.𝒟²ᶻᴺ[1,1] * diffMatrix.𝒟ᶻᴺ[1,iter+1]/diffMatrix.𝒟ᶻᴺ[1,1])

    #     diffMatrix.𝒟²ᶻᴺ[n,iter]   = (diffMatrix.𝒟²ᶻᴺ[n,iter] + 
    #                             -1.0 * diffMatrix.𝒟²ᶻᴺ[n,n] * diffMatrix.𝒟ᶻᴺ[n,iter]/diffMatrix.𝒟ᶻᴺ[n,n])
    # end

    # diffMatrix.𝒟²ᶻᴺ[1,1] = 0.0
    # diffMatrix.𝒟²ᶻᴺ[n,n] = 0.0

    # @. diffMatrix.𝒟ᶻᴺ[1,1:end] = 0.0
    # @. diffMatrix.𝒟ᶻᴺ[n,1:end] = 0.0

    setBCs!(diffMatrix, params, "dirchilet")
    setBCs!(diffMatrix, params, "neumann"  )
    
    kron!( Op.𝒟ᶻᴰ  ,  Iʸ , diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟²ᶻᴰ ,  Iʸ , diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟⁴ᶻᴰ ,  Iʸ , diffMatrix.𝒟⁴ᶻᴰ )

    kron!( Op.𝒟ᶻᴺ  ,  Iʸ , diffMatrix.𝒟ᶻᴺ )
    kron!( Op.𝒟²ᶻᴺ ,  Iʸ , diffMatrix.𝒟²ᶻᴺ)

    kron!( Op.𝒟ʸ   ,  diffMatrix.𝒟ʸ  ,  Iᶻ ) 
    kron!( Op.𝒟²ʸ  ,  diffMatrix.𝒟²ʸ ,  Iᶻ )
    kron!( Op.𝒟⁴ʸ  ,  diffMatrix.𝒟⁴ʸ ,  Iᶻ ) 

    kron!( Op.𝒟ʸᶻᴰ   ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟ᶻᴰ  )
    kron!( Op.𝒟ʸ²ᶻᴰ  ,  diffMatrix.𝒟ʸ  ,  diffMatrix.𝒟²ᶻᴰ )
    kron!( Op.𝒟²ʸ²ᶻᴰ ,  diffMatrix.𝒟²ʸ ,  diffMatrix.𝒟²ᶻᴰ )

    return nothing
end



