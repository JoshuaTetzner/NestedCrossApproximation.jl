using BEAST
using FastBEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
using Random

h = 0.025
λ = 20h
k = 2π / λ
##
Γ = meshsphere(1.0, h)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(3)
ttree = KMeansTree(space.pos, 2; minvalues=100)
stree = ttree
tree = BlockTree(ttree, stree)

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
    #ntasks=,
);

##
@time A = assemble(op, space, space);
x = rand(ComplexF64, length(space))
##

estimate_reldifference(wnca, A)
##

testcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        MaximumValue(),
        TreeMimicryPivoting(space.pos, space.pos, tree.trialcluster),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        TreeMimicryPivoting(space.pos, space.pos, tree.testcluster),
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
    #ntasks=1,
);
