using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra

h = 0.04
λ = 10
k = 2π / λ
##
Γ = meshsphere(1.0, h)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, space, 0.01; minvaluestest=100, minvaluestrial=100)
#ttree = KMeansTree(space.pos, 2; minvalues=100)
#stree = ttree
#tree = BlockTree(ttree, stree)
#
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
    ntasks=10,
);

##
@time A = assemble(op, space, space);
x = rand(ComplexF64, length(space))
##

norm(wnca * x - A * x) / norm(A * x)
