module NCABlockSparse

using NestedCrossApproximation
using BEAST
using BlockSparseMatrices

include("nearblockassembler.jl")

function assembleblockprimer(assembler)
    return BEAST.assembleblock_primer(
        NestedCrossApproximation.operator(assembler),
        NestedCrossApproximation.testspace(assembler),
        NestedCrossApproximation.trialspace(assembler);
        quadstrat=NestedCrossApproximation.quadstrategy(assembler),
    )
end

end
