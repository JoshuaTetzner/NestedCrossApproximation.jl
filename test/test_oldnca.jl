using BEAST
using FastBEAST
using NestedCrossApproximation
using ClusterTrees
using LinearAlgebra
using CompScienceMeshes

λ = 3e8
k = 2 * π / λ
gamma = im * k
Γ = meshicosphere(28, 1.0)
op = Maxwell3D.singlelayer(; gamma=gamma, alpha=1.0, beta=0.0)
space = raviartthomas(Γ)
#A = assemble(op, space, space)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=200, maxlevel=100))
##
testcomp = NestedCrossApproximation.ButtomUpCompressor(
    iACA(
        LRF.MaximumValue(Bool[]),
        NestedCrossApproximation.IACAPivoting2(tree, space.pos),
        NestedCrossApproximation.IncompleteNormEstimator(Float64[], Float64(0.0)),
    ),
    nothing,
)
trialcomp = NestedCrossApproximation.ButtomUpCompressor(
    NestedCrossApproximation.iACA(
        NestedCrossApproximation.IACAPivoting2(tree, space.pos),
        LRF.MaximumValue(Bool[]),
        NestedCrossApproximation.IncompleteNormEstimator(Float64[], Float64(0.0)),
    ),
    nothing,
)
bu = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    maxrank=200,
    multithreading=true,
    tol=1e-5,
    η=1.0,
)

##
lrbb = NestedCrossApproximation.lrbmat(bu)
lrbfm = NestedCrossApproximation.lrbmat(A, bu)
##
norm(lrbfm - lrbb) / norm(lrbfm)