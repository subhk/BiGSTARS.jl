# ==============================================================================
# eig_solver.jl - A Unified Interface for Generalized Eigenvalue Problems
# ==============================================================================

using VectorInterface: MinimalSVec, MinimalMVec, MinimalVec

# ==============================================================================
# Data Structures
# ==============================================================================

"""
    SolverConfig

Configuration parameters for eigenvalue solvers.

# Fields
- `method::Symbol`: Solver method (:Arnoldi, :Arpack, :Krylov)
- `σ₀::Float64`: Initial shift value
- `which::Symbol`: Which eigenvalues to find (:LM, :LR, :SR, etc.)
- `nev::Int`: Number of eigenvalues to compute
- `maxiter::Int`: Maximum iterations
- `tol::Float64`: Convergence tolerance
- `sortby::Symbol`: Sort eigenvalues by (:R, :I, :M)
- `n_tries::Int`: Number of retry attempts
- `Δσ₀::Float64`: Initial shift increment
- `incre::Float64`: Increment growth factor
- `ϵ::Float64`: Successive eigenvalue tolerance
- `krylovdim::Int`: Krylov subspace dimension (Krylov method only)
"""
@kwdef struct SolverConfig
    method::Symbol = :Krylov
    σ₀::Float64
    which::Symbol = :LM
    nev::Int = 1
    maxiter::Int = 300
    tol::Float64 = 1e-12
    sortby::Symbol = :M
    n_tries::Int = 8
    Δσ₀::Float64 = 0.2
    incre::Float64 = 1.2
    ϵ::Float64 = 1e-5
    krylovdim::Int = 200
end

"""
    ConvergenceHistory

Tracks convergence information during eigenvalue computation.
"""
mutable struct ConvergenceHistory
    attempts::Vector{Float64}           # Shift values attempted
    converged::Vector{Bool}             # Whether each attempt converged
    eigenvalues::Vector{ComplexF64}     # Eigenvalues found at each attempt
    errors::Vector{String}              # Error messages for failed attempts
    final_shift::Float64                # Final successful shift
    total_iterations::Int               # Total iterations across all attempts
    
    ConvergenceHistory() = new(Float64[], Bool[], ComplexF64[], String[], 0.0, 0)
end

"""
    SolverResults

Contains results from eigenvalue computation.

# Fields
- `eigenvalues::Vector{ComplexF64}`: Computed eigenvalues
- `eigenvectors::Matrix{ComplexF64}`: Corresponding eigenvectors
- `converged::Bool`: Whether computation converged
- `method_used::Symbol`: Method that produced the result
- `final_shift::Float64`: Final shift value used
- `iterations::Int`: Total iterations
- `solve_time::Float64`: Wall-clock time for solution
- `history::ConvergenceHistory`: Detailed convergence information
"""
struct SolverResults
    eigenvalues::Vector{ComplexF64}
    eigenvectors::Matrix{ComplexF64}
    converged::Bool
    method_used::Symbol
    final_shift::Float64
    iterations::Int
    solve_time::Float64
    history::ConvergenceHistory
end

"""
    EigenSolver

Main solver object for generalized eigenvalue problems Ax = λBx.

# Fields
- `A::AbstractMatrix`: Left-hand side matrix
- `B::AbstractMatrix`: Right-hand side matrix
- `config::SolverConfig`: Solver configuration
- `results::Union{SolverResults, Nothing}`: Latest results (if available)
"""
mutable struct EigenSolver
    A::AbstractMatrix
    B::AbstractMatrix
    config::SolverConfig
    results::Union{SolverResults, Nothing}
    
    function EigenSolver(A, B, config::SolverConfig)
        # Validate inputs
        size(A) == size(B) || throw(DimensionMismatch("Matrices A and B must have the same size"))
        size(A, 1) == size(A, 2) || throw(ArgumentError("Matrices must be square"))
        
        new(A, B, config, nothing)
    end
end

# Convenience constructor
function EigenSolver(A, B; σ₀::Float64, kwargs...)
    config = SolverConfig(; σ₀=σ₀, kwargs...)
    return EigenSolver(A, B, config)
end

# ==============================================================================
# Utility Functions
# ==============================================================================

function wrapvec(v, ::Val{mode}) where {mode}
    return mode === :vector ? v :
           mode === :inplace ? MinimalMVec(v) :
           mode === :outplace ? MinimalSVec(v) :
           mode === :mixed ? MinimalSVec(v) :
           throw(ArgumentError("invalid mode ($mode)"))
end

unwrapvec(v::MinimalVec) = v.vec
unwrapvec(v) = v

# Julia version compatibility
if VERSION < v"1.9"
    stack(f, itr) = mapreduce(f, hcat, itr)
    stack(itr) = reduce(hcat, itr)
end

"""
    sort_eigenvalues!(λ, Χ, by::Symbol; rev::Bool=true)

Sort eigenvalues and eigenvectors in-place.
- `:R` → real part
- `:I` → imaginary part  
- `:M` → magnitude
"""
function sort_eigenvalues!(λ::Vector, Χ::Matrix, by::Symbol; rev::Bool=true)
    sortfun = by == :R ? real : by == :I ? imag : abs
    idx = sortperm(λ, by=sortfun, rev=rev)
    return λ[idx], Χ[:, idx]
end

# function construct_linear_map(A_shifted, B)
#     # This function should be defined based on your specific linear map construction
#     # For now, assuming it returns a linear map for the shifted problem
#     return LinearMap(x -> A_shifted \ (B * x), size(A_shifted, 1))
# end

# ==============================================================================
# Core Solver Methods
# ==============================================================================

"""
    solve_arnoldi_single(op, σ; config)

Single Arnoldi solve without retry logic. Expects a prebuilt shift-and-invert
operator `op` for the current shift `σ` to avoid per-call allocations.
"""
function solve_arnoldi_single(op, σ::Float64; config::SolverConfig)
    decomp, history = partialschur(op;
                                   nev=config.nev,
                                   tol=config.tol,
                                   restarts=config.maxiter,
                                   which=config.which)
    
    μ, Χ = partialeigen(decomp)
    λ = @. 1.0 / μ + σ
    
    return sort_eigenvalues!(λ, Χ, config.sortby; rev=true)
end

"""
    solve_arpack_single(A, B, σ; config)

Single ARPACK solve without retry logic.
"""
function solve_arpack_single(A, B, σ::Float64; config::SolverConfig)
    λ, Χ, info = Arpack.eigs(A, B;
                             nev=config.nev,
                             sigma=σ,
                             which=config.which,
                             maxiter=config.maxiter,
                             tol=config.tol,
                             check=0)
    
    return sort_eigenvalues!(λ, Χ, config.sortby; rev=true)
end

"""
    solve_krylov_single(op, σ; config)

Single KrylovKit solve without retry logic. Expects a prebuilt shift-and-invert
operator `op` for the current shift `σ` to avoid per-call allocations.
"""
function solve_krylov_single(op, σ::Float64; config::SolverConfig)
    λinv, Χ, info = eigsolve(op,
                             rand(ComplexF64, size(op, 1)),
                             config.nev,
                             config.which;
                             maxiter=config.maxiter,
                             krylovdim=config.krylovdim,
                             tol=config.tol,
                             verbosity=0)
    
    λ = @. 1.0 / λinv + σ
    
    return sort_eigenvalues!(λ, stack(unwrapvec, Χ), config.sortby; rev=true)
end

# ==============================================================================
# Main Solver Interface
# ==============================================================================

"""
    solve!(solver::EigenSolver; verbose::Bool=true)

Solve the generalized eigenvalue problem with adaptive shift selection.

# Returns
- `solver.results`: Updated with solution results

# Example
```julia
solver = EigenSolver(A, B; σ₀=1.0, method=:Arnoldi, nev=5)
solve!(solver)
λ, Χ = get_results(solver)
```
"""
function solve!(solver::EigenSolver; verbose::Bool=true)
    config = solver.config
    history = ConvergenceHistory()
    
    # Reusable buffers to avoid per-attempt allocations
    A_shifted = solver.A + solver.B                 # establishes union sparsity pattern
    temp      = Vector{eltype(A_shifted)}(undef, size(solver.A, 1))
    
    # Generate shift attempts
    Δσs_up = [config.Δσ₀ * config.incre^(i-1) * abs(config.σ₀) for i in 1:config.n_tries]
    Δσs_dn = [-δ for δ in Δσs_up]
    σ_attempts = [config.σ₀; (config.σ₀ + δ for δ in vcat(Δσs_up, Δσs_dn))...]
    
    λ_prev = nothing
    last_λ = ComplexF64[]
    last_Χ = zeros(ComplexF64, size(solver.A, 1), 0)
    start_time = time()
    
    for (i, σ) in enumerate(σ_attempts)
        push!(history.attempts, σ)
        
        verbose && @printf("(attempt %2d/%2d) trying σ = %.6f with %s\n", 
                          i, length(σ_attempts), σ, config.method)
        
        try
            # Dispatch to appropriate solver
            λ, Χ = if config.method == :Arnoldi
                A_shifted .= solver.A .- σ .* solver.B
                op = construct_linear_map(factorize(A_shifted), solver.B, temp)
                solve_arnoldi_single(op, σ; config=config)
            elseif config.method == :Arpack  
                solve_arpack_single(solver.A, solver.B, σ; config=config)
            elseif config.method == :Krylov
                A_shifted .= solver.A .- σ .* solver.B
                op = construct_linear_map(factorize(A_shifted), solver.B, temp)
                solve_krylov_single(op, σ; config=config)
            else
                throw(ArgumentError("Unknown method: $(config.method)"))
            end
            
            push!(history.converged, true)
            push!(history.eigenvalues, λ[1])
            push!(history.errors, "")
            last_λ = ComplexF64.(λ)
            last_Χ = ComplexF64.(Χ)
            
            verbose && @printf("  ✓ converged: λ₁ = %.6f + %.6fi\n", real(λ[1]), imag(λ[1]))
            
            # Check successive convergence
            if λ_prev !== nothing && abs(λ[1] - λ_prev) < config.ϵ
                verbose && @printf("  ✓ successive eigenvalues converged: |Δλ| = %.2e < %.2e\n", 
                                  abs(λ[1] - λ_prev), config.ϵ)
                
                history.final_shift = σ
                solve_time = time() - start_time
                
                solver.results = SolverResults(
                    λ, Χ, true, config.method, σ, i, solve_time, history
                )
                return solver
            end
            
            λ_prev = λ[1]
            
        catch err
            push!(history.converged, false)
            push!(history.eigenvalues, NaN + 0im)
            push!(history.errors, string(err))
            
            verbose && @warn "  ✗ failed at σ = $σ: $err"
        end
    end

    # A single converged attempt is still a usable result when the caller does
    # not require the successive-shift stability check to pass.
    last_success = findlast(history.converged)
    if last_success !== nothing
        history.final_shift = history.attempts[last_success]
        solve_time = time() - start_time
        solver.results = SolverResults(
            last_λ, last_Χ,
            true, config.method, history.final_shift, length(σ_attempts),
            solve_time, history
        )
        return solver
    end
    
    # If we get here, all attempts failed
    solve_time = time() - start_time
    solver.results = SolverResults(
        ComplexF64[], zeros(ComplexF64, 0, 0), false, config.method, 
        config.σ₀, length(σ_attempts), solve_time, history
    )
    
    error("❌ All $(length(σ_attempts)) attempts failed to converge for method $(config.method)")
end

"""
    get_results(solver::EigenSolver)

Extract eigenvalues and eigenvectors from solved system.

# Returns
- `(λ, Χ)`: Eigenvalues and eigenvectors

# Throws
- `ArgumentError`: If solver hasn't been solved or didn't converge
"""
function get_results(solver::EigenSolver)
    isnothing(solver.results) && throw(ArgumentError("Solver has not been run. Call solve!(solver) first."))
    !solver.results.converged && throw(ArgumentError("Solver did not converge. Check solver.results for details."))
    
    return solver.results.eigenvalues, solver.results.eigenvectors
end

# ==============================================================================
# Comparison and Analysis Tools
# ==============================================================================

"""
    compare_methods!(solver::EigenSolver; methods=[:Arnoldi, :Arpack, :Krylov], verbose=true)

Compare multiple methods on the same problem.

# Returns
- `Dict{Symbol, SolverResults}`: Results for each method
"""
function compare_methods!(solver::EigenSolver; 
                         methods::Vector{Symbol}=[:Arnoldi, :Arpack, :Krylov],
                         verbose::Bool=true)
    
    original_config = solver.config
    results = Dict{Symbol, Union{SolverResults, Nothing}}()
    
    verbose && println("🔍 Comparing methods for eigenvalue problem...")
    verbose && println("   Problem size: $(size(solver.A))")
    verbose && println("   Initial shift: $(original_config.σ₀)")
    verbose && println("   Methods: $methods")
    verbose && println("   " * "="^50)
    
    for method in methods
        verbose && println("\n Testing method: $method")
        
        # Update config for this method
        new_config = SolverConfig(
            method = method,
            σ₀ = original_config.σ₀,
            which = original_config.which,
            nev = original_config.nev,
            maxiter = original_config.maxiter,
            tol = original_config.tol,
            sortby = original_config.sortby,
            n_tries = original_config.n_tries,
            Δσ₀ = original_config.Δσ₀,
            incre = original_config.incre,
            ϵ = original_config.ϵ,
            krylovdim = original_config.krylovdim
        )
        solver.config = new_config
        
        try
            solve!(solver; verbose=false)
            results[method] = solver.results
            
            if solver.results.converged
                λ₁ = solver.results.eigenvalues[1]
                verbose && @printf("  ✅ Success: λ₁ = %.6f + %.6fi (%.3fs, %d attempts)\n",
                                  real(λ₁), imag(λ₁), solver.results.solve_time, 
                                  length(solver.results.history.attempts))
            else
                verbose && println("  ❌ Failed to converge")
            end
            
        catch err
            results[method] = nothing
            verbose && println("  ❌ Error: $err")
        end
    end
    
    # Restore original config
    solver.config = original_config
    
    # Summary
    if verbose
        println("\n Summary:")
        successful_methods = [m for (m, r) in results if !isnothing(r) && r.converged]
        if !isempty(successful_methods)
            fastest = minimum(m -> results[m].solve_time, successful_methods)
            fastest_idx = findfirst(m -> results[m].solve_time == fastest, successful_methods)
            fastest_method = successful_methods[fastest_idx]
            println("   Successful methods: $successful_methods")
            @printf("   Fastest method: %s (%.3fs)\n", fastest_method, fastest)

            # Check consistency
            eigenvals = [results[m].eigenvalues[1] for m in successful_methods]
            if length(eigenvals) > 1
                max_diff = maximum(abs(λ - eigenvals[1]) for λ in eigenvals)
                @printf("   Max eigenvalue difference: %.2e\n", max_diff)
            end
        else
            println("   ❌ No methods converged successfully")
        end
        println("   " * "="^50)
    end
    
    return results
end

"""
    print_summary(solver::EigenSolver)

Print a summary of solver results.
"""
function print_summary(solver::EigenSolver)
    if isnothing(solver.results)
        println("❌ Solver has not been run")
        return
    end
    
    r = solver.results
    println("EigenSolver Results Summary")
    println("   " ^ 40)
    println("   Method: $(r.method_used)")
    println("   Converged: $(r.converged ? "✅ Yes" : "❌ No")")
    println("   Final shift: $(r.final_shift)")
    @printf("   Total time: %.3fs\n", r.solve_time)
    println("   Attempts: $(length(r.history.attempts))")

    if r.converged
        nλ    = length(r.eigenvalues)
        idx_w = length(string(nλ))             # index‐column width

        # Pre‑format each eigenvalue once so we know the widest line
        fmt(v) = @sprintf("% .6f %+.6fi", real(v), imag(v))
        valstr = [fmt(λ) for λ in r.eigenvalues]
        val_w  = maximum(length, valstr)       # eigenvalue column width

        println("   ├─ Eigenvalues ($nλ) " * "─"^max(10, 16 - idx_w))
        @printf("   │ %*s │ %-*s │\n", idx_w, "i", val_w, "λ (Re  Im·i)")
        println("   │" * "─"^(idx_w + 2) * "┼" * "─"^(val_w + 2) * "│")

        for (i, s) in enumerate(valstr)
            @printf("   │ %*d │ %-*s │\n", idx_w, i, val_w, s)
        end

        println("   └" * "─"^(idx_w + val_w + 5) * "┘")
    end
    println("   " ^ 40)

end

# ==============================================================================
# Convenience Functions and Backwards Compatibility
# ==============================================================================

"""
    solve_eigenvalue_problem(A, B; method=:Arnoldi, σ₀, verbose=false, kwargs...)

Convenience function that creates a solver and immediately solves it.
Maintains backwards compatibility with the original interface.
"""
function solve_eigenvalue_problem(A, B; method::Symbol=:Arnoldi, σ₀::Float64,
                                  verbose::Bool=false, kwargs...)
    config = SolverConfig(; method=method, σ₀=σ₀, kwargs...)
    solver = EigenSolver(A, B, config)
    solve!(solver; verbose=verbose)
    return get_results(solver)
end

# Legacy convenience functions
solve_arnoldi(A, B; σ₀::Float64, verbose::Bool=false, kwargs...) =
    solve_eigenvalue_problem(A, B; method=:Arnoldi, σ₀=σ₀, verbose=verbose, kwargs...)

solve_arpack(A, B; σ₀::Float64, verbose::Bool=false, kwargs...) =
    solve_eigenvalue_problem(A, B; method=:Arpack, σ₀=σ₀, verbose=verbose, kwargs...)

solve_krylov(A, B; σ₀::Float64, verbose::Bool=false, kwargs...) =
    solve_eigenvalue_problem(A, B; method=:Krylov, σ₀=σ₀, verbose=verbose, kwargs...)

# ==============================================================================
# Documentation and Help
# ==============================================================================

"""
    get_method_info(method::Symbol)

Get detailed information about a specific solver method.
"""
function get_method_info(method::Symbol)
    info = Dict(
        :Arnoldi => """
        Arnoldi Method (ArnoldiMethod.jl)
        
        Best for: General sparse matrices, moderate problem sizes
        Strengths: Reliable, good theoretical foundation
        Key parameters: which, nev, maxiter, tol
        
        Typical usage:
        solver = EigenSolver(A, B; σ₀=1.0, method=:Arnoldi, nev=5, tol=1e-10)
        """,
        
        :Arpack => """
        ARPACK Method (Arpack.jl)  
        
        Best for: Large sparse matrices, production use
        Strengths: Battle-tested, widely used, efficient for large problems
        Key parameters: which, nev, maxiter, tol
        
        Typical usage:
        solver = EigenSolver(A, B; σ₀=1.0, method=:Arpack, which=:LM, maxiter=500)
        """,
        
        :Krylov => """
        Krylov Method (KrylovKit.jl)
        
        Best for: Modern problems, flexible interface
        Strengths: Clean interface, good for research
        Key parameters: which, maxiter, krylovdim
        
        Typical usage:
        solver = EigenSolver(A, B; σ₀=1.0, method=:Krylov, krylovdim=150)
        """
    )
    
    return get(info, method, "❌ Unknown method: $method\nAvailable: :Arnoldi, :Arpack, :Krylov")
end

"""
    show_example_usage()

Display comprehensive usage examples.
"""
function show_example_usage()
    println("""
    eig_solver.jl Usage Examples
    =====================================
    
    # Basic usage with the new interface:
    using LinearAlgebra
    A, B = rand(100, 100), I  # Your matrices here
    
    # Method 1: Object-oriented interface (recommended)
    solver = EigenSolver(A, B; σ₀=1.0, method=:Arnoldi, nev=5)
    solve!(solver)
    λ, Χ = get_results(solver)
    print_summary(solver)
    
    # Method 2: Functional interface (backwards compatible)  
    λ, Χ = solve_eigenvalue_problem(A, B; method=:Arpack, σ₀=1.0, nev=3)
    
    # Method 3: Compare multiple methods
    solver = EigenSolver(A, B; σ₀=1.0, nev=3)
    results = compare_methods!(solver; methods=[:Arnoldi, :Arpack])
    
    # Method 4: Custom configuration
    config = SolverConfig(
        method = :Krylov,
        σ₀ = 2.5, 
        which = :LR,
        nev = 10,
        maxiter = 500,
        tol = 1e-14,
        krylovdim = 200
    )
    solver = EigenSolver(A, B, config)
    solve!(solver)
    
    # Get help on methods:
    println(get_method_info(:Arnoldi))
    """)
end

# ==============================================================================
# Usage Examples (outside module)
# ==============================================================================

"""
Quick start example:

```julia
using BiGSTARS

# Create test matrices
n = 100
A = rand(n, n); A = A + A'  # Make symmetric
B = Matrix(I, n, n)         # Identity matrix

# Solve with Arnoldi method
solver = EigenSolver(A, B; σ₀=1.0, method=:Arnoldi, nev=5)
solve!(solver)
λ, Χ = get_results(solver)
print_summary(solver)

# Compare all methods
results = compare_methods!(solver)

# Show detailed information
show_example_usage()
println(get_method_info(:Arpack))
```
"""
