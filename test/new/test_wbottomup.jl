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
    ntasks=1,
);

##
H2Trees.values(tree.testcluster, 285)
##
A = assemble(op, space, space)
##
t = 11
Ft = [42, 770, 771, 772, 773, 775, 779]

isnear = NestedCrossApproximation.isnear(k)

for s in Ft
    println(isnear(tree.testcluster, tree.trialcluster, t, s))
end
##
x = rand(ComplexF64, size(wnca, 2))
norm(A * x - wnca * x) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(adjoint(A) * x - adjoint(wnca) * x) / norm(adjoint(A) * x)
