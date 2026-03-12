module NCABEAST

using NestedCrossApproximation
using BEAST

include("kernelmatrix.jl")

function NestedCrossApproximation.defaultfarmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.DoubleNumQStrat(2, 3)
end

function NestedCrossApproximation.defaultmatrixdata(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return BEAST.defaultquadstrat(operator, testspace, trialspace)
end

#NestedCrossApproximation.wavenumber(operator::BEAST.IntegralOperator) = imag(operator.gamma)

end
