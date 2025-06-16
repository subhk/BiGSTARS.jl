# using LazyGrids
# using BlockArrays
# using Printf
# using StaticArrays
# #using Interpolations
# using SparseArrays
# using SparseMatrixDicts
# using SpecialFunctions
# using FillArrays
# using Parameters
# using Test
# using MAT
# using BenchmarkTools

# using Serialization
# #using Pardiso
# using Arpack
# using LinearMaps

using ArnoldiMethod: partialschur, partialeigen, LR, LI, LM

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end


function construct_linear_map(A, B)
    a = ShiftAndInvert( factorize(A), B, Vector{eltype(A)}(undef, size(A,1)) )
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end


function Eigs_Arnoldi(𝓛, ℳ; σ::Float64, maxiter::Int, which)

    decomp, history = partialschur(construct_linear_map(𝓛 - σ*ℳ, ℳ), 
                                    nev=20, 
                                    tol=1e-12, 
                                    restarts=maxiter, 
                                    which=which)

    println(history)

    λₛ⁻¹, Χ = partialeigen(decomp)
    λₛ = @. 1.0 / λₛ⁻¹ + σ
    return λₛ, Χ
end


function EigSolver_shift_invert_arnoldi(𝓛, ℳ; σ₀::Float64, maxiter, which)
    #maxiter::Int = 20
    try 
        σ = 1.10σ₀
        @printf "sigma: %f \n" real(σ) 
        λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
        return λₛ, Χ
    catch error
        try 
            σ = 1.05σ₀
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
            return λₛ, Χ
        catch error
            try 
                σ = 0.99σ₀
                @printf "(second didn't work) sigma: %f \n" real(σ) 
                λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                return λₛ, Χ
            catch error
                try
                    σ = 0.95σ₀
                    @printf "(third didn't work) sigma: %f \n" real(σ) 
                    λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                    return λₛ, Χ
                catch error
                    try
                        σ = 0.93σ₀
                        @printf "(fourth didn't work) sigma: %f \n" real(σ) 
                        λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                        return λₛ, Χ
                    catch error
                        try
                            σ = 0.90σ₀
                            @printf "(fifth didn't work) sigma: %f \n" real(σ) 
                            λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                            return λₛ, Χ   
                        catch error
                            try
                                σ = 0.85σ₀
                                @printf "(sixth didn't work) sigma: %f \n" real(σ) 
                                λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                            catch error
                                σ = 0.80σ₀
                                @printf "(seventh didn't work) sigma: %f \n" real(σ)
                                λₛ, Χ = Eigs_Arnoldi(𝓛, ℳ; σ=σ, maxiter=maxiter, which=which)
                                return λₛ, Χ
                            end
                        end            
                    end
                end
            end
        end    
    end
end



# function EigSolver_shift_invert_arnoldi(𝓛, ℳ; σ₀::Float64)
#     maxiter::Int = 20
#     try 
#         σ = 1.15σ₀
#         @printf "sigma: %f \n" real(σ) 
#         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#         return λₛ, Χ
#     catch error
#         try
#             σ = 1.10σ₀
#             @printf "(first didn't work) sigma: %f \n" real(σ) 
#             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             return λₛ, Χ
#         catch error
#             try
#                 σ = 1.05σ₀
#                 @printf "(second didn't work) sigma: %f \n" real(σ)
#                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                 return λₛ, Χ
#             catch error
#                 try
#                     σ = 0.99σ₀
#                     @printf "(third didn't work) sigma: %f \n" real(σ)
#                     λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                     return λₛ, Χ
#                 catch error
#                     try
#                         σ = 0.95σ₀
#                         @printf "(fourth didn't work) sigma: %f \n" real(σ)
#                         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                         return λₛ, Χ 
#                     catch error
#                         try
#                             σ = 0.90σ₀
#                             @printf "(fifth didn't work) sigma: %f \n" real(σ)
#                             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                             return λₛ, Χ
#                         catch error
#                             try
#                                 σ = 0.85σ₀
#                                 @printf "(sixth didn't work) sigma: %f \n" real(σ)
#                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                 return λₛ, Χ
#                             catch error
#                                 try
#                                     σ = 0.82σ₀
#                                     @printf "(seventh didn't work) sigma: %f \n" real(σ) 
#                                     λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                     return λₛ, Χ
#                                 catch error
#                                     try
#                                         σ = 0.78σ₀
#                                         @printf "(eighth didn't work) sigma: %f \n" real(σ) 
#                                         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                         return λₛ, Χ
#                                     catch error
#                                         try
#                                             σ = 0.75σ₀
#                                             @printf "(ninth didn't work) sigma: %f \n" real(σ) 
#                                             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                             return λₛ, Χ
#                                         catch error
#                                             try
#                                                 σ = 0.72σ₀
#                                                 @printf "(tenth didn't work) sigma: %f \n" real(σ) 
#                                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 return λₛ, Χ
#                                             catch error
#                                                 σ = 0.69σ₀
#                                                 @printf "(eleventh didn't work) sigma: %f \n" real(σ)
#                                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 return λₛ, Χ
#                                             end    
#                                         end   
#                                     end
#                                 end    
#                             end
#                         end                    
#                     end          
#                 end    
#             end
#         end
#     end
# end

# function EigSolver_shift_invert_arnoldi2(𝓛, ℳ; σ₀::Float64)
#     maxiter::Int = 20
#     try 
#         σ = 0.90σ₀
#         @printf "sigma: %f \n" real(σ) 
#         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#         return λₛ, Χ
#     catch error
#         try
#             σ = 0.87σ₀
#             @printf "(first didn't work) sigma: %f \n" real(σ) 
#             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             return λₛ, Χ
#         catch error
#             try
#                 σ = 0.84σ₀
#                 @printf "(second didn't work) sigma: %f \n" real(σ)
#                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                 return λₛ, Χ
#             catch error
#                 try
#                     σ = 0.81σ₀
#                     @printf "(third didn't work) sigma: %f \n" real(σ)
#                     λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                     return λₛ, Χ
#                 catch error
#                     try
#                         σ = 0.78σ₀
#                         @printf "(fourth didn't work) sigma: %f \n" real(σ)
#                         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                         return λₛ, Χ 
#                     catch error
#                         try
#                             σ = 0.75σ₀
#                             @printf "(fifth didn't work) sigma: %f \n" real(σ)
#                             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                             return λₛ, Χ
#                         catch error
#                             try
#                                 σ = 0.70σ₀
#                                 @printf "(sixth didn't work) sigma: %f \n" real(σ)
#                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                 return λₛ, Χ
#                             catch error
#                                 try
#                                     σ = 0.65σ₀
#                                     @printf "(seventh didn't work) sigma: %f \n" real(σ) 
#                                     λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                     return λₛ, Χ
#                                 catch error
#                                     try
#                                         σ = 0.60σ₀
#                                         @printf "(eighth didn't work) sigma: %f \n" real(σ) 
#                                         λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                         return λₛ, Χ
#                                     catch error
#                                         try
#                                             σ = 0.55σ₀
#                                             @printf "(ninth didn't work) sigma: %f \n" real(σ) 
#                                             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                             return λₛ, Χ
#                                         catch error
#                                             try
#                                                 σ = 0.50σ₀
#                                                 @printf "(tenth didn't work) sigma: %f \n" real(σ) 
#                                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 return λₛ, Χ
#                                             catch error
#                                                 σ = 0.45σ₀
#                                                 @printf "(eleventh didn't work) sigma: %f \n" real(σ)
#                                                 λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#                                                 return λₛ, Χ
#                                             end    
#                                         end   
#                                     end
#                                 end    
#                             end
#                         end                    
#                     end          
#                 end    
#             end
#         end
#     end
# end