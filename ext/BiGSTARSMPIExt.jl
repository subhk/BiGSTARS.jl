module BiGSTARSMPIExt

using BiGSTARS
using BiGSTARS: _to_csr, _csr_row_block, SolverResults, ConvergenceHistory,
                sort_eigenvalues!, _filter_physical_modes
using MPI
using PetscWrap
using SlepcWrap
using SparseArrays
using LinearAlgebra: norm

# Implementations added in Tasks 6–9.

end # module
