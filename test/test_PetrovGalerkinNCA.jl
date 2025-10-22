using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using Test

##

Γ = meshsphere(1.0, 0.08)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
A = assemble(op, space, space)
##

@time h2mat, tp, sp = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=false,
)
