using NestedCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using FastBEAST
using ClusterTrees
using LinearAlgebra
using ThreadsX

Γ = meshsphere(1.0, 0.1)

space = raviartthomas(Γ)

λ = 5
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)

tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree; η=1.0)

A = assemble(op, space, space)
@views function fct(B, x, y)
    return B[:, :] = A[x, y]
end
##
PetrovGalerkinMCA(op, space, space;)

##

pos = [@SVector rand(3) for i in 1:10]

pivstrat = LRF.ModifiedFillDistance(pos)

pivstrat(Vector(1:5))

pivstrat()
