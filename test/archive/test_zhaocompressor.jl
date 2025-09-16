using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using Test
using Random
##

λ = 15
k = 2 * π / λ
Γ = meshsphere(1.0, 0.06)#CompScienceMeshes.read_gmsh_mesh(pwd() * "/test/rafale3268.msh")
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(1)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=200))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree; η=1.0)
##
A = assemble(op, space, space)
##
@views farblkassembler = BEAST.blockassembler(op, space, space)
@views function farassembler(Z, tdata, sdata)
    @views store(v, m, n) = (Z[m, n] += v)
    return farblkassembler(tdata, sdata, store)
end
##
compressor = NestedCrossApproximation.ZhaoCompressor()
@time h2matz = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=compressor,
    trialcompressor=compressor,
    multithreading=true,
    tol=1e-4,
)
lrbA = NestedCrossApproximation.lrbmat(A, h2matz)
lrbh2matz = NestedCrossApproximation.lrbmat(h2matz)
norm(lrbA - lrbh2matz) / norm(lrbA)

##

FastBEAST.cluster_link(tree)
fars
sfars = NestedCrossApproximation.testfars(length(tree.nodes), fars)
h2matz.nestedtestbases[42]
##
value(tree, 19)
value(tree, 18)
value(tree, 17)
sfars = NestedCrossApproximation.testfars(length(tree.nodes), fars)

sfars[19]
##
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
@time h2mat = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    multithreading=true,
)
##
A = assemble(op, space, space)

lrbh2mat = NestedCrossApproximation.lrbmat(h2mat)
lrbh2matz = NestedCrossApproximation.lrbmat(h2matz)
lrbA = NestedCrossApproximation.lrbmat(A, h2mat)
##
norm(lrbA - lrbh2mat) / norm(lrbA)
norm(lrbA - lrbh2matz) / norm(lrbA)
##
for blk in h2matz.couplingmatrices
    τ = value(tree, blk.row_basis)
    σ = value(tree, blk.col_basis)

    if (!ClusterTrees.haschildren(tree, blk.row_basis)) &&
        (norm(lrbA[τ, σ] - lrbh2matz[τ, σ]) / norm(lrbA[τ, σ])) > 5e-1
        println(blk.row_basis, ", ", blk.col_basis)
    end
end
##
fars[end]
fars
FastBEAST.cluster_link(tree)
h2matz.nestedtestbases[42].T
collect(ClusterTrees.leaves(tree, 1))
##
sfars = NestedCrossApproximation.testfars(length(tree.nodes), fars)
collect(ClusterTrees.leaves(tree, 51))
##
node = 18
sfars[18]
ch1 = collect(ClusterTrees.children(tree, node))
badnode = 47
ch1 = collect(ClusterTrees.children(tree, badnode))
collect(ClusterTrees.leaves(tree, badnode))

##
h2matz.testtransfermatrices[6][65].T[1]
