using BEAST
using FastBEAST
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using CompScienceMeshes

h = 0.01
λ = 20h
k = 2π / λ
Γ = meshrectangle(1.0, 1.0, h)
space = raviartthomas(Γ)
Random.seed!(3)
ttree = KMeansTree(space.pos, 2; minvalues=100)
stree = ttree
tree = BlockTree(ttree, stree)

##
isnear = NestedCrossApproximation.isnear(k)
tfarnodes = NestedCrossApproximation.farinteractions(ttree, stree; isnear=isnear)
sfarnodes = NestedCrossApproximation.farinteractions(stree, ttree; isnear=isnear)

tfarnodes == sfarnodes
tfarnodes[6]
sfarnodes[388]
