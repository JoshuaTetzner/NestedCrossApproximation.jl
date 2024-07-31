using SparseArrays
using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation 


Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer(;)
cxd0 = lagrangecxd0(Γ);

tree=create_tree(cxd0.pos, KMeansTreeOptions())
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = computeinteractions(blktree)

@views farblkassembler = BEAST.blockassembler(
    op, cxd0, cxd0
)
@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkassembler(tdata,sdata,store)
end
##
fars
##
@time test_fars = row_pivot_selection(
    tree,
    tree,
    fars,
    farassembler,
    scalartype(op), 
);
##
##
@time test_fars = NestedCrossApproximation.sparserow_pivot_selection(
    tree,
    tree,
    fars,
    farassembler,
    scalartype(op),
);