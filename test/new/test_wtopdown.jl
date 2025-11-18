using BEAST
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
##

λ = 0.6
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.03)
##
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, space, 0.01; minvaluestest=200, minvaluestrial=200)
##
testcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(space.pos, space.pos),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MimicryPivoting(space.pos, space.pos),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
@time wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    space,
    space,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=50,
);
##
x = rand(ComplexF64, size(wnca, 2))
@time wnca * x;

##
leveledbases = Int[]
for level in H2Trees.levels(tree.testcluster)
    levelbases = 0
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        if haskey(wnca.nestedtestbases, t)
            levelbases += 1
        end
    end
    push!(leveledbases, levelbases)
end
leveledbases

leveledtransfer = Int[]
for level in H2Trees.levels(tree.testcluster)
    leveltransfer = 0
    for t in H2Trees.LevelIterator(tree.testcluster, level)
        if haskey(wnca.testtransfermatrices[level], t)
            leveltransfer += 1
        end
    end
    push!(leveledtransfer, leveltransfer)
end
leveledbases
leveledtransfer
##
x = rand(ComplexF64, size(wnca, 2))
##
A = assemble(op, space, space)

norm(A * x - wnca * x) / norm(A * x)

##

using FastBEAST

FastBEAST.estimate_reldifference(wnca, A; tol=1e-5)
##
using BlockSparseMatrices
using OhMyThreads
blocks = [rand(100, 100) for _ in 1:1000]
values = [collect((i * 100):(i * 100 + 99)) for i in 1:1000]
@time nearinteractionsp = BlockSparseMatrix(
    blocks, values, values, (100099, 100099); scheduler=DynamicScheduler()
)
@time nearinteractions = BlockSparseMatrix(blocks, values, values, (100099, 100099))

x = rand(size(nearinteractionsp, 2))
@time nearinteractionsp * x;
@time nearinteractions * x;
