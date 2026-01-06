module NCABEAST

using NestedCrossApproximation
using BEAST

include("kernelmatrix.jl")

function NestedCrossApproximation.defaultfarquadstrat(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.DoubleNumQStrat(2, 3)
end

function NestedCrossApproximation.defaultnearquadstrat(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

NestedCrossApproximation.wavenumber(operator::BEAST.IntegralOperator) = imag(operator.gamma)

end
