function NestedCrossApproximation.assemble(
    assembler::NestedCrossApproximation.BlockBEASTNearInteractionsAssembler,
    values,
    nearvalues,
)
    return _blocksparseassemble(assembler, values, nearvalues)
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    f.matrix[m, n] += v
    return nothing
end

function assembleblock(
    ::NestedCrossApproximation.BlockBEASTNearInteractionsAssembler,
    matrixblock,
    tdata,
    sdata,
    blkasm,
)
    blkasm(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function blockassembler(
    assembler::NestedCrossApproximation.BlockBEASTNearInteractionsAssembler;
    primer=assembleblockprimer(assembler),
)
    return BEAST.blockassembler(
        NestedCrossApproximation.operator(assembler),
        NestedCrossApproximation.testspace(assembler),
        NestedCrossApproximation.trialspace(assembler);
        quadstrat=NestedCrossApproximation.quadstrategy(assembler),
        primer=primer,
    )
end

function _blocksparseassemble(
    assembler::NestedCrossApproximation.AbstractNearInteractionsAssembler{T},
    values,
    nearvalues;
) where {T}
    blocks = Vector{Matrix{T}}(undef, length(values))
    addblocks!(blocks, values, nearvalues)
    @assert length(blocks) == length(values) == length(nearvalues)

    primer = assembleblockprimer(assembler)

    Threads.@threads for i in eachindex(values)
        assembleblock(
            assembler,
            blocks[i],
            values[i],
            nearvalues[i],
            blockassembler(assembler; primer=primer),
        )
    end

    return BlockSparseMatrix(blocks, values, nearvalues, size(assembler))
end

function addblocks!(blocks::Vector{Matrix{T}}, values, nearvalues) where {T}
    @assert length(blocks) == length(values) == length(nearvalues)
    Threads.@threads for i in eachindex(values)
        blocks[i] = zeros(T, length(values[i]), length(nearvalues[i]))
    end
    return blocks
end
