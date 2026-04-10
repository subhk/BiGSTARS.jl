using Printf

# Custom struct for nicely formatted eigenvalue output
struct EigenvalueDisplay
    λ::Complex
    idx::Int
end

# Custom display for EigenvalueDisplay
function Base.show(io::IO, ::MIME"text/plain", ev::EigenvalueDisplay)
    r, i = real(ev.λ), imag(ev.λ)
    if iszero(i)
        @printf(io, "%3d │ % .6e          ", ev.idx, r)
    else
        sign_str = i ≥ 0 ? "+" : ""
        @printf(io, "%3d │ % .6e %s%.6eim", ev.idx, r, sign_str, i)
    end
end

"""
    print_evals(λs::AbstractVector{<:Number}; n::Int=length(λs), by::Symbol=:abs)

Pretty-print the top `n` eigenvalues from the list `λs`, optionally sorted by `:real`, `:imag`, or `:abs`.
"""
function print_evals(λs::Vector{<:Complex})
    n = length(λs) 
    λs_sorted = λs #sort(λs, by=abs, rev=true)
    println("Top $n eigenvalues (sorted):")
    println("Idx │ Real Part     Imag Part")
    println("────┼──────────────────────────────")
    for p in n:-1:1
        show(stdout, "text/plain", EigenvalueDisplay(λs_sorted[p], p))
        println()  # ensure newline after each
    end
end


"""
    λs_sorted, χ_sorted = sort_evals(λs, χ, "R"; sorting="lm")
    Sort the eigenvalues `λs` and corresponding eigenvectors `χ` based on the specified criterion 
    (`"M"` for magnitude, `"I"` for imaginary part, or `"R"` for real part).
    The `sorting` argument determines the order: `"lm"` for descending order.
"""
function sort_evals(λs::AbstractVector, χ::AbstractMatrix, which::String; sorting::String="lm")
    @assert which in ("M", "I", "R")
    by_func = Dict(
        "M" => abs,
        "I" => imag,
        "R" => real
    )[which]

    idx = sortperm(λs, by=by_func, rev=(sorting == "lm"))
    return λs[idx], χ[:, idx]
end



"""
    sort_evals(λ, Χ, by::Symbol; rev::Bool=true)

Sort eigenvalues `λ` and corresponding eigenvectors `Χ` by:
- `:R` → real part
- `:I` → imaginary part
- `:M` → magnitude (abs)

Set `rev=true` for descending (default), `false` for ascending.
"""
function sort_evals(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true)
    sortfun = by == :R ? real : by == :I ? imag : abs
    idx = sortperm(λ, by=sortfun, rev=rev)
    return λ[idx], Χ[:, idx]
end


function remove_evals(λs, χ, lower, higher, which)
    
    @assert which ∈ ["M", "I", "R"]

    if which == "I" # imaginary part
        arg = findall( (lower .≤ imag(λs)) .& (imag(λs) .≤ higher) )
    end

    if which == "R" # real part
        arg = findall( (lower .≤ real(λs)) .& (real(λs) .≤ higher) )
    end
    
    if which == "M" # absolute magnitude
        arg = findall( (lower .≤ abs.(λs)) .& (abs.(λs) .≤ higher) )
    end
    
    χ  = χ[:,arg]
    λs = λs[arg]

    return λs, χ

end
