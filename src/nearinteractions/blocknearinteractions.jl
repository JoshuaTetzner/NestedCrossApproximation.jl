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
