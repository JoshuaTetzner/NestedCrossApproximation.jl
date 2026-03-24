using BEAST
using AdaptiveCrossApproximation
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using LinearAlgebra
##

λ = 2
k = 2 * pi / λ
ηₗ = 1.0
ηₕ = 4.0
Γ = meshsphere(1.0, 0.05)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
ttree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
tree = BlockTree(ttree, ttree)
##
@time h2 = NestedCrossApproximation.PetrovGalerkinNCA2(op, space, space, tree;);
##
testcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.iACAPivoting(space.pos, space.pos),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(1e-4)
        ),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.iACAPivoting(space.pos, space.pos),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(1e-4)
        ),
    ),
    nothing,
)
@time h2 = NestedCrossApproximation.PetrovGalerkinNCA2(
    op, space, space, tree; testcompressor=testcompressor, trialcompressor=trialcompressor
);

##
testcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.iACAPivoting(space.pos),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(1e-4)
        ),
    ),
    nothing,
)
##
AdaptiveCrossApproximation.iACAPivoting(space.pos)
##
@time fm = assemble(op, space, space);
##
x = rand(ComplexF64, size(fm, 2))

norm(h2 * x - fm * x) / norm(fm * x)
##
NestedCrossApproximation.AbstractKernelMatrix(k, k, k)
##
buffer = NestedCrossApproximation.allocate_buffer(ComplexF64, (1, 1), (2, 2);)
reverse(NestedCrossApproximation.allocate_buffer(ComplexF64, (1, 1), (2, 2);))
##

iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> H2Trees.isnear)(tree)
testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), 3))

nodes = []
for s in collect(iterator(ttree, ttree, 3))
    append!(nodes, H2Trees.values(ttree, s))
end
nodes
