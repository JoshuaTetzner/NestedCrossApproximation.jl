"""
    testbases(compressor, farmatrix, tree, fardata; scheduler, maxrank, pbar)
    trialbases(compressor, farmatrix, tree, fardata; scheduler, maxrank, pbar)

Build one side's nested bases and transfer matrices, returning
`(basis_store, transfer_store, pivots)`.

This is the contract every compressor implements. For each `fardata` type it supports, a
compressor must provide:

  - `testbases` and `trialbases` as above;
  - `_basesworkload(compressor, tree, fardata)`, the dry-run weight used to size the
    assembly progress bar. Those methods live in utils/workload.jl, next to the
    `_farworkload` passes they are built from and to each other, since the point of each
    is to total exactly what its own traversal increments.

Implementations: [`TopDown`](@ref) and [`BottomUp`](@ref) for `FarData`,
[`BottomUp`](@ref) for `DirectionalData` (wideband), [`BatchedBottomUp`](@ref) for
`FarData`.
"""
function testbases end

@doc (@doc testbases) function trialbases end

# Wideband (directional) assembly is implemented for `BottomUp` only: `TopDown` has no
# DirectionalData `testbases`/`trialbases`, and `BatchedBottomUp`'s geometric
# pivotselector has no directional far-field selection. Checked before any work so an
# unsupported compressor reports itself, instead of surfacing as a MethodError from
# progress-bar sizing. The accepting `::BottomUp` method lives in bottomup.jl, so the
# file that implements wideband is the file that opts into it.
function _check_wideband_compressor(compressor, side::Symbol)
    return throw(
        ArgumentError(
            "Wideband (directional) NCA is only supported with BottomUp; got " *
            "$(typeof(compressor)) as $side compressor.",
        ),
    )
end
