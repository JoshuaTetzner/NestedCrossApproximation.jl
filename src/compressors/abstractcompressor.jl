"""
    testbases(compressor, farmatrix, tree, fardata; ntasks, maxrank, pbar)
    trialbases(compressor, farmatrix, tree, fardata; ntasks, maxrank, pbar)

Build the nested bases and transfer matrices of the test or the trial side.

Interface of the compressors: [`TopDown`](@ref), [`BottomUp`](@ref), and
[`BatchedBottomUp`](@ref) implement both functions for [`FarData`](@ref), [`BottomUp`](@ref)
also for [`DirectionalData`](@ref).

# Arguments

  - `compressor`: compressor that selects the pivots
  - `farmatrix`: kernel matrix for the evaluation of far-field entries
  - `tree`: block tree
  - `fardata`: far interactions of the side that is compressed

# Keyword Arguments

  - `ntasks::Integer`: number of parallel tasks; `ntasks = 1` runs on the calling task
  - `maxrank::Int`: maximum number of pivots of a cluster
  - `pbar`: progress bar

# Returns

  - `basisstore`: leaf bases, see [`BasisStore`](@ref)
  - `transferstore`: transfer matrices, see [`TransferStore`](@ref)
  - `pivots`: pivots of each cluster (or of each cluster and direction)

# Notes

A new compressor additionally implements `_basesworkload(compressor, tree, fardata)`, which
sizes the progress bar.
"""
function testbases end

@doc (@doc testbases) function trialbases end

# Directional far fields are implemented for `BottomUp` only, which overrides this method.
function _check_wideband_compressor(compressor, side::Symbol)
    return throw(
        ArgumentError(
            "Wideband (directional) NCA is only supported with BottomUp; got " *
            "$(typeof(compressor)) as $side compressor.",
        ),
    )
end
