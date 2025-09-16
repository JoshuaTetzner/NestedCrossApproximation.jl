
function NestedCrossApproximation.AbstractKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    primer=BEAST.assembleblock_primer(operator, testspace, trialspace; quadstrat=quadstrat),
)
    return NestedCrossApproximation.BEASTKernelMatrix{scalartype(operator)}(
        operator, testspace, trialspace, primer, quadstrat
    )
end

function NestedCrossApproximation.BEASTKernelMatrix(
    operator::BEAST.IntegralOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space;
    quadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    primer=BEAST.assembleblock_primer(operator, testspace, trialspace; quadstrat=quadstrat),
)
    return NestedCrossApproximation.BEASTKernelMatrix{scalartype(operator)}(
        operator, testspace, trialspace, primer, quadstrat
    )
end
struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    f.matrix[m, n] += v
    return nothing
end

function assembleblock(
    ::NestedCrossApproximation.BEASTKernelMatrix, matrixblock, tdata, sdata, blkasm
)
    blkasm(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function blockassembler(assembler::NestedCrossApproximation.BEASTKernelMatrix)
    return BEAST.blockassembler(
        assembler.operator,
        assembler.testspace,
        assembler.trialspace;
        quadstrat=assembler.quadstrat,
        primer=assembler.primer,
    )
end

function (blk::NestedCrossApproximation.BEASTKernelMatrix)(matrixblock, tdata, sdata)
    return assembleblock(blk, matrixblock, tdata, sdata, blockassembler(blk))
end
