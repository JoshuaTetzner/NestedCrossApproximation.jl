using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using AdaptiveCrossApproximation
using LinearAlgebra

λ = 0.48
k = 2 * pi / λ
Γ = meshsphere(1.0, 0.024)

op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
ttree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
tree = BlockTree(ttree, ttree)
##
function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=5.0)
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return false) : (return true)
    end
end
issnear(treea, treeb, nodea, nodeb) = isnear(k, treea, treeb, nodea, nodeb)

##
tol = 1e-3
testcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.iACAPivoting(space.pos, space.pos),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(tol)
        ),
    ),
    nothing,
)
trialcompressor = NestedCrossApproximation.TopDownCompressor(
    AdaptiveCrossApproximation.iACA(
        AdaptiveCrossApproximation.iACAPivoting(space.pos, space.pos),
        AdaptiveCrossApproximation.MaximumValue(),
        AdaptiveCrossApproximation.FNormExtrapolator(
            AdaptiveCrossApproximation.iFNormEstimator(tol)
        ),
    ),
    nothing,
)
wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op,
    space,
    space,
    tree;
    isnear=issnear,
    lfcompressor=AdaptiveCrossApproximation.ACA(; convergence=FNormEstimator(tol)),
    testcompressor=testcompressor,
    trialcompressor=trialcompressor,
    maxrank=100,
);

##
x = rand(ComplexF64, size(wnca, 2))
wnca * x

A = assemble(op, space, space)
##
norm(A * x - wnca * x) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(adjoint(A) * x - adjoint(wnca) * x) / norm(adjoint(A) * x)

##
islf = NestedCrossApproximation.islf(NestedCrossApproximation.wavenumber(op))
dtree = NestedCrossApproximation.𝒟tree(H2Trees.halfsize(tree.testcluster), imag(op.gamma))
values, nearvalues, fars, dirs, lfvalues, lffarvalues = NestedCrossApproximation.directionalitneractions(
    tree, dtree, islf, issnear
)
dirs
@time tf = NestedCrossApproximation.testfars(dtree, tree, fars, dirs, islf)
fars
lfvalues

tf
##
tf[3]

using StaticArrays
dir = NestedCrossApproximation.direction(
    SVector(0.0, 0.0, 0.0) - SVector(1.0, 1.0, 1.0), dtree, 4
)

dtree.level
