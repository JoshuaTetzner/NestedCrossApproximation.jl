using CompScienceMeshes
using StaticArrays

λ = 0.5
k = 2 * pi / λ

##

Γ1 = meshrectangle(1.0, 1.0, 0.1)
Γ2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(4.0, 0.0, 0.0))

##
X1 = raviartthomas(Γ1)
X2 = raviartthomas(Γ2)

##
ttree = TwoNTree(X1, 0.0; minvalues=100)
stree = TwoNTree(X2, 0.0; minvalues=100)
tree = BlockTree(ttree, stree)

##
complexisnear(ta, tb, a, b) =
    NestedCrossApproximation.isnear(k, ta, tb, a, b; ηₗ=1.0, ηₕ=4.0)
values, nearvalues = H2Trees.nearinteractions(
    tree; isnear=complexisnear, extractselfvalues=false
)

##
functor = H2Trees.WellSeparatedIterator(; isnear=(tree) -> complexisnear)
iterator = functor(tree)
##
levelvalues = Vector{Int}[]
levelfars = Vector{Vector{Int}}[]
for level in H2Trees.levels(ttree)
    values = collect(H2Trees.LevelIterator(ttree, level))
    farvalues = Vector{Vector{Int}}(undef, length(values))
    for (i, node) in enumerate(values)
        println(collect(iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), node)))
        farvalues[i] = collect(
            iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), node)
        )
    end
    push!(levelvalues, values)
    push!(levelfars, farvalues)
end

levelvalues
levelfars
##
ttree2 = create_tree(X1.pos, BoxTreeOptions(; maxlevel=1, nmin=500))
stree2 = create_tree(X2.pos, BoxTreeOptions(; maxlevel=1, nmin=500))
blktree = ClusterTrees.BlockTrees.BlockTree(ttree2, stree2)
nears, hffars, lffars = NestedCrossApproximation.computeinteractionshf(
    blktree, k; ηₗ=1.0, ηₕ=4.0
)

sort(nears)
