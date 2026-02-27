
function defaultfarquadstrat(operator, testspace, trialspace) end
function defaultnearquadstrat(operator, testspace, trialspace) end

mutable struct TopDownCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function TopDownCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function TopDownCompressor(;
    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
)
    return TopDownCompressor(factorization, representor)
end

struct BottomUpCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function BottomUpCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function BottomUpCompressor(;
    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
)
    return BottomUpCompressor(factorization, representor)
end

function GalerkinNCA(op, space;) end

function PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace;
    tree=TwoNTree(testspace, trialspace, 2 / 2^20; mintestvalues=100, mintrialvalues=100),
    testcompressor=BottomUpCompressor(),
    trialcompressor=BottomUpCompressor(),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    isnear=defaultisnear(op),
)
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    values, nearvalues = nearinteractions(tree; isnear=isnear)
    blocks = zeros.(eltype(nearmatrix), length.(values), length.(nearvalues))
    @tasks for i in eachindex(blocks)
        @set ntasks = ntasks
        nearmatrix(blocks[i], values[i], nearvalues[i])
    end
    nears = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )
    testfars, trialfars = farinteractions(tree, isnear)
    nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix, testfars, tree, buffer, isnear; ntasks=ntasks
    )
    nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix, trialfars, tree, buffer, isnear; ntasks=ntasks
    )
    couplingmatrices = assemble_couplingmatrices(
        farmatrix, testpivots, trialpivots, testfars, trialfars; ntasks=ntasks
    )
    return nothing
end
