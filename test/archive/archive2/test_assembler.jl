using BEAST
#using BlockSparseMatrices
using NestedCrossApproximation
#using H2Trees
using CompScienceMeshes
##
λ = 2
k = 2 * pi / λ

Γ = meshsphere(1.0, 0.08)
op = Maxwell3D.singlelayer(; wavenumber=k)

space = raviartthomas(Γ)

@time x = NestedCrossApproximation.BEASTKernelMatrix(op, space, space);

eltype(x)
blk = zeros(ComplexF64, 10, 10)
x(blk, 1:10, 1:10)
blk
##
struct BlockStoreFunctor{M}
    matrix::M
end

function (f::BlockStoreFunctor)(v, m, n)
    f.matrix[m, n] += v
    return nothing
end

struct BlockBEASTMatrix{
    T,OperatorType,TestSpaceType,TrialSpaceType,PrimerType,QuadStratType
} <: AbstractMatrix{T}
    operator::OperatorType
    testspace::TestSpaceType
    trialspace::TrialSpaceType
    primer::PrimerType
    quadstrat::QuadStratType
    function BlockBEASTMatrix{T}(
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

function BlockBEASTMatrix(operator, space)
    return BlockBEASTMatrix{scalartype(operator)}(
        operator,
        space,
        space,
        BEAST.assembleblock_primer(op, space, space),
        BEAST.defaultquadstrat(operator, space, space),
    )
end

function assembleblock(::BlockBEASTMatrix, matrixblock, tdata, sdata, blkasm)
    blkasm(tdata, sdata, BlockStoreFunctor(matrixblock))
    return nothing
end

function blockassembler(assembler::BlockBEASTMatrix)
    return BEAST.blockassembler(
        assembler.operator,
        assembler.testspace,
        assembler.trialspace;
        quadstrat=assembler.quadstrat,
        primer=assembler.primer,
    )
end

function (blk::BlockBEASTMatrix)(tdata, sdata, matrixblock)
    return assembleblock(blk, matrixblock, tdata, sdata, blockassembler(blk))
end

##
λ = 2
k = 2 * pi / λ

Γ = meshsphere(1.0, 0.08)
op = Maxwell3D.singlelayer(; wavenumber=k)

space = raviartthomas(Γ)

ttree = H2Trees.TwoNTree(space.pos, 0.2)
tree = H2Trees.BlockTree(ttree, ttree)
##
function assembleDJ(op, space, tree)
    nearassembler = NestedCrossApproximation.BlockBEASTNearInteractionsAssembler{
        scalartype(op)
    }(
        op, space, space, BEAST.defaultquadstrat(op, space, space)
    )

    return nearinteractions = nearassembler(tree, H2Trees.isnear)
end

function myassembler(op, space, tree)
    M = BlockBEASTMatrix(op, space)

    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=H2Trees.isnear, extractselfvalues=false
    )

    blocks = Vector{Matrix{scalartype(op)}}(undef, length(values))
    #NestedCrossApproximation.addblocks!(blocks, values, nearvalues)

    Threads.@threads for i in eachindex(values)
        blk = zeros(scalartype(op), length(values[i]), length(nearvalues[i]))
        M(values[i], nearvalues[i], blk)
        blocks[i] = blk
    end
end

##
@time assembleDJ(op, space, tree);
@time myassembler(op, space, tree);
