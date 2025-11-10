using ProgressMeter

abstract type AbstractNearInteractionsAssembler{T} end

function Base.size(assembler::AbstractNearInteractionsAssembler, dim::Int)
    return size(assembler)[dim]
end

function operator(assembler::AbstractNearInteractionsAssembler)
    return assembler.operator
end

function trialspace(assembler::AbstractNearInteractionsAssembler)
    return assembler.trialspace
end

function testspace(assembler::AbstractNearInteractionsAssembler)
    return assembler.testspace
end

function quadstrategy(assembler::AbstractNearInteractionsAssembler)
    return assembler.quadstrat
end

# to use BlockBEASTNearInteractionsAssembler load the extension H2BlockSparse which is dependent
# on BEAST and BlockSparseMatrices
struct BlockBEASTNearInteractionsAssembler{
    T,OperatorType,TestSpaceType,TrialSpaceType,QuadStratType
} <: AbstractNearInteractionsAssembler{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    quadstrat::QuadStratType

    function BlockBEASTNearInteractionsAssembler{T}(
        operator, testspace, trialspace, quadstrat
    ) where {T}
        return new{
            T,typeof(operator),typeof(testspace),typeof(trialspace),typeof(quadstrat)
        }(
            operator, testspace, trialspace, quadstrat
        )
    end
end

function Base.size(assembler::BlockBEASTNearInteractionsAssembler)
    return (length(assembler.testspace), length(assembler.trialspace))
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree, isnear; verbose::Bool=false, kwargs...
)
    return assembler(tree, H2Trees.treetrait(tree), isnear; verbose=verbose, kwargs...)
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree, ::H2Trees.isBlockTree, isnear; verbose::Bool=false, kwargs...
)
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false, kwargs...
    )
    return assemble(assembler, values, nearvalues, verbose)
end

function assemble(
    assembler::BlockBEASTNearInteractionsAssembler, values, nearvalues, verbose::Bool
)
    return _blocksparseassemble(assembler, values, nearvalues, verbose)
end

struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    f.matrix[m, n] += v
    return nothing
end

function assembleblock(
    ::BlockBEASTNearInteractionsAssembler, matrixblock, tdata, sdata, blkasm
)
    blkasm(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function assembleblockprimer(assembler)
    return BEAST.assembleblock_primer(
        operator(assembler),
        testspace(assembler),
        trialspace(assembler);
        quadstrat=quadstrategy(assembler),
    )
end

function blockassembler(
    assembler::BlockBEASTNearInteractionsAssembler; primer=assembleblockprimer(assembler)
)
    return BEAST.blockassembler(
        operator(assembler),
        testspace(assembler),
        trialspace(assembler);
        quadstrat=quadstrategy(assembler),
        primer=primer,
    )
end

function _blocksparseassemble(
    assembler::AbstractNearInteractionsAssembler{T}, values, nearvalues, verbose::Bool;
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
