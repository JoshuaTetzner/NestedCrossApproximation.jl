using CompScienceMeshes
using H2Trees

Γ = meshsphere(1.0, 0.05)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, 0.0; minvalues=200)
blktree = BlockTree(tree, tree)

##

values, nearvalues = H2Trees.nearinteractions(blktree; extractselfvalues=false)
functor = H2Trees.WellSeparatedIterator(; isnear=(tree) -> complexisnear)
iterator = functor(tree)

for node in iterator(H2Trees.testtree(blktree), H2Trees.testtree(blktree), 4)
end

x = Vector(1:10)

Threads.@threads for i in x
    println(i)
end
