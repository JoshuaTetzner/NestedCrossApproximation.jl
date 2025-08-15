module NCABlockSparse

using BEAST
using BlockSparseMatrices

include("nearblockassembler.jl")

function assembleblockprimer(assembler)
    return BEAST.assembleblock_primer(
        operator(assembler),
        testspace(assembler),
        trialspace(assembler);
        quadstrat=quadstrategy(assembler),
    )
end

end
