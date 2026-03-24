using BEAST
using FastBEAST
using H2Trees
using ParallelKMeans
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
using Random
using Test

h = 0.025
Γ = meshsphere(1.0, h)
##
_, _, h = edgeinfo(Γ)
λ = 20h     # Wavelength
k = 2 * π / λ   # Wavenumber
op = Maxwell3D.singlelayer(; wavenumber=k, alpha=-im * k, beta=zero(λ) * im)
space = raviartthomas(Γ)
Random.seed!(1)
ttree = KMeansTree(space.pos, 2; minvalues=100)
stree = KMeansTree(space.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

##

testcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        MaximumValue(),
        TreeMimicryPivoting(space.pos, space.pos, tree.trialcluster),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
);
trialcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        TreeMimicryPivoting(space.pos, space.pos, tree.testcluster),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
);
@time wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    space,
    space,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=50,
    ntasks=10,
);
