struct BEASTKernelMatrix{
    T,OperatorType,TestSpaceType,TrialSpaceType,PrimerType,QuadStratType
} <: AbstractKernelMatrix{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    primer::PrimerType
    quadstrat::QuadStratType
    function BEASTKernelMatrix{T}(
        operator, testspace, trialspace, primer, quadstrat
    ) where {T}
        return new{
            T,
            typeof(operator),
            typeof(testspace),
            typeof(trialspace),
            typeof(primer),
            typeof(quadstrat),
        }(
            operator, testspace, trialspace, primer, quadstrat
        )
    end
end

function Base.size(M::BEASTKernelMatrix, dim=nothing)
    if dim === nothing
        return (length(M.testspace), length(M.trialspace))
    elseif dim == 1
        return length(M.testspace)
    elseif dim == 2
        return length(M.trialspace)
    else
        error("dim must be either 1 or 2")
    end
end

AdaptiveCrossApproximation.nextrc!(buf, A::BEASTKernelMatrix, i, j) = A(buf, i, j)
