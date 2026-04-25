#══════════════════════════════════════════════════════════════════════════════#
#  Solve: wavenumber loop driver with optional thread parallelism             #
#══════════════════════════════════════════════════════════════════════════════#

"""
    solve(cache, k_values; sigma_0, method=:Krylov, parallel=false, kwargs...)

Solve the eigenvalue problem for each wavenumber in `k_values`.
Returns a `Vector{SolverResults}`.

Uses **in-place assembly** to reuse the main dense matrix workspace. When
`parallel=true`, one pre-allocated workspace per thread is created. Problems
with derived variables still allocate per wavenumber while rebuilding `H(k)`.

## Keyword arguments

- `sigma_0::Float64` — initial shift for the eigenvalue solver (required)
- `method::Symbol` — solver method: `:Krylov` (default), `:Arnoldi`, or `:Arpack`
- `parallel::Bool` — if `true`, solve wavenumbers in parallel using `Threads.@threads`.
  Start Julia with multiple threads (`julia -t auto`) to benefit from this.
- `verbose::Bool` — print solver progress (default `false`)

## Example

```julia
cache = discretize(prob)
k_values = range(0.1, 5.0, length=50)

# Sequential (1 workspace, reused for all wavenumbers):
results = solve(cache, k_values; sigma_0=0.02)

# Parallel (1 workspace per thread, reused across wavenumbers):
results = solve(cache, k_values; sigma_0=0.02, parallel=true)
```
"""
function solve end

"""
    solve(cache; sigma_0, kwargs...) -> Vector{SolverResults}

Solve a problem with no FourierTransformed direction (pure 1D/2D, no wavenumber sweep).
"""
function solve(cache::DiscretizationCache;
               sigma_0::Float64, method::Symbol=:Krylov,
               parallel::Bool=false, verbose::Bool=false, kwargs...)
    return solve(cache, [0.0]; sigma_0=sigma_0, method=method,
                 parallel=parallel, verbose=verbose, kwargs...)
end

function solve(cache::DiscretizationCache, k_values::AbstractVector;
               sigma_0::Float64, method::Symbol=:Krylov,
               parallel::Bool=false, verbose::Bool=false, kwargs...)
    n = length(k_values)
    results = Vector{SolverResults}(undef, n)

    failed_result = SolverResults(
        ComplexF64[], zeros(ComplexF64, 0, 0), false, method,
        sigma_0, 0, 0.0, ConvergenceHistory()
    )

    if parallel
        # One workspace per thread — allocated once, reused across all wavenumbers
        n_threads = Threads.nthreads()
        workspaces = [allocate_workspace(cache) for _ in 1:n_threads]

        Threads.@threads for i in 1:n
            tid = Threads.threadid()
            ws = workspaces[tid]
            results[i] = _solve_inplace(ws, cache, Float64(k_values[i]),
                                        sigma_0, method, verbose, failed_result; kwargs...)
        end
    else
        # Single workspace, reused for every wavenumber
        ws = allocate_workspace(cache)
        for i in 1:n
            results[i] = _solve_inplace(ws, cache, Float64(k_values[i]),
                                        sigma_0, method, verbose, failed_result; kwargs...)
        end
    end

    return results
end

"""Solve a single wavenumber using the pre-allocated main matrix workspace."""
function _solve_inplace(ws::AssemblyWorkspace, cache::DiscretizationCache,
                        k_val::Float64, sigma_0::Float64, method::Symbol,
                        verbose::Bool, failed_result::SolverResults; kwargs...)
    # In-place assembly reuses ws.A and ws.B for the main system matrices.
    assemble!(ws, cache, k_val)

    # EigenSolver accepts AbstractMatrix, so dense matrices work fine
    solver = EigenSolver(ws.A, ws.B; σ₀=sigma_0, method=method, kwargs...)
    try
        solve!(solver; verbose=verbose)
        return solver.results
    catch e
        verbose && @warn "Failed at k = $k_val: $e"
        return failed_result
    end
end
