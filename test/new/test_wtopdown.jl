using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
##

λ = 0.6
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.03)

op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
ttree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
tree = BlockTree(ttree, ttree)

##
testcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(space.pos, space.pos),
        FNormExtrapolator(iFNormEstimator(1e-4)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    iACA(
        MimicryPivoting(space.pos, space.pos),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-4)),
    ),
    nothing,
)
wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    space,
    space,
    tree;
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=100,
);

x = rand(ComplexF64, size(wnca, 2))

##
A = assemble(op, space, space)

norm(A * x - wnca * x) / norm(A * x)
