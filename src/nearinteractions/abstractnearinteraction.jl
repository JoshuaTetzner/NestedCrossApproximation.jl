abstract type AbstractNearInteractionsAssembler{T} end

function (assembler::AbstractNearInteractionsAssembler)(tree, isnear; kwargs...)
    return assembler(tree, H2Trees.treetrait(tree), isnear; kwargs...)
end

function (assembler::AbstractNearInteractionsAssembler)(
    tree, ::H2Trees.isBlockTree, isnear; kwargs...
)
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false, kwargs...
    )
    return assemble(assembler, values, nearvalues)
end

function assemble(assembler::AbstractNearInteractionsAssembler, values, nearvalues) end

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
