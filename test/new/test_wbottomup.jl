using BEAST
using H2Trees
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using LinearAlgebra
##

λ = 0.5
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.05)
##
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, space, 0.01; minvaluestest=200, minvaluestrial=200)
##
testcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        MaximumValue(),
        MimicryPivoting(space.pos, space.pos),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.BottomUpCompressor(
    iACA(
        MimicryPivoting(space.pos, space.pos),
        MaximumValue(),
        FNormExtrapolator(iFNormEstimator(1e-3)),
    ),
    nothing,
)

trialcompressor.lrf.convergence.estimator.tol = 1e-4

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
A = assemble(op, space, space)

##
x = rand(ComplexF64, size(wnca, 2))
norm(A * x - wnca * x) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(adjoint(A) * x - adjoint(wnca) * x) / norm(adjoint(A) * x)
