using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using Test
using Random
##

λ = 5
k = 2 * π / λ
Γ = meshicosphere(20, 1.0)#CompScienceMeshes.read_gmsh_mesh(pwd() * "/test/rafale3268.msh")
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(2)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=100))
##
println("ButtomUp")

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

@time h2mat_bup = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=true,
    tol=1e-4,
)
##
h2mt = h2mat_bup
h2sl = PetrovGalerkinNCA{ComplexF64}(
    h2mt.tree,
    h2mt.nearinteractions,
    h2mt.nestedtestbases,
    h2mt.nestedtrialbases,
    h2mt.testtransfermatrices,
    h2mt.trialtransfermatrices,
    h2mt.couplingmatrices,
    h2mt.fars,
    h2mt.dim,
    false,
)
##
x = rand(ComplexF64, 108000)
##
using Base.Threads
Threads.nthreads() = 8
h2mt * x;
@time h2mt * x;
@time h2mt * x;

Threads.nthreads() = 1
h2sl * x;
@time h2sl * x;
@time h2sl * x;
##
A = rand(ComplexF64, 108000, 108000)
@time A * x;
##

println("TopDown")
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
@time h2mat_tdown = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=true,
    tol=1e-4,
)

##
lrbA = NestedCrossApproximation.lrbmat(A, h2mat_bup)
lrbh2matz = NestedCrossApproximation.lrbmat(h2mat_bup)
norm(lrbA - lrbh2matz) / norm(lrbA)
