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

Base.size(M::BEASTKernelMatrix) = (length(M.testspace), length(M.trialspace))
