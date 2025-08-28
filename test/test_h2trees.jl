using ParallelKMeans
using H2Trees
using BEAST
using CompScienceMeshes
using NestedCrossApproximation
using ClusterTrees
##

Γ = meshsphere(1.0, 0.1)

ttree = H2Trees.KMeansTree(vertices(Γ), 2; minvalues=100)
tree = BlockTree(ttree, ttree)
values, nearvalues = H2Trees.nearinteractions(tree; extractselfvalues=false)
values
nearvalues

leafs = collect(H2Trees.leaves(ttree))
##
ttree2 = NestedCrossApproximation.create_tree(
    vertices(Γ), NestedCrossApproximation.KMeansTreeOptions(; nmin=100)
)

leafs2 = collect(ClusterTrees.leaves(ttree2))
##
blktree = ClusterTrees.BlockTrees.BlockTree(ttree2, ttree2)
nears, fars = NestedCrossApproximation.computeinteractions(blktree; η=2.0)

sort(nears)

mynears = unique!([near[1] for near in nears])

ttree.nodes
ttree2.nodes

for mynear in mynears
    if !(mynear in leafs2)
        println(mynear)
    end
end

collect(children(ttree2, 45))
