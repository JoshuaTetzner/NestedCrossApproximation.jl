using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using CompScienceMeshes
using StaticArrays

λ = 0.1
k = 2π / λ

Γ = meshrectangle(1.0, 1.0, 0.01)
islf = NestedCrossApproximation.islf(k)
isnear = NestedCrossApproximation.isnear(k)

##
ttree = KMeansTree(Γ.vertices, 2; minvalues=10)
stree = ttree
tree = BlockTree(ttree, stree)

testfardata = NestedCrossApproximation.directionaltestfars(tree; islf=islf, isnear=isnear);
trialfardata = NestedCrossApproximation.directionaltrialfars(tree; islf=islf, isnear=isnear)

testfardata.𝓔
testfardata.F
##
node = 9
println("child ", testfardata.𝓔[node])
println("childmap ", testfardata.𝓔map[node])
testfardata.F[node]
p = H2Trees.parent(ttree, node)
println("parent ", testfardata.𝓔[p])
println("parentmap ", testfardata.𝓔map[p])
pp = H2Trees.parent(ttree, p)
println("parent ", testfardata.𝓔[pp])
println("parentmap ", testfardata.𝓔map[pp])

##
for level in H2Trees.levels(ttree)
    println(
        length(
            findall(
                x -> x != [], testfardata.F[collect(H2Trees.LevelIterator(ttree, level))]
            ),
        ),
    )
end
##
ttree = TwoNTree(Γ.vertices, 0.25; minvalues=400)
stree = TwoNTree(Γ.vertices, 0.25; minvalues=400)
tree = BlockTree(ttree, stree)

testfardata = NestedCrossApproximation.directionaltestfars(tree; islf=islf, isnear=isnear)
trialfardata = NestedCrossApproximation.directionaltrialfars(tree; islf=islf, isnear=isnear)

##
testfardata.𝓔
testfardata.inherited𝓔
##

a = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

mapreduce(vcat, a) do b
    b[1] == 1 && return b
    return Int[]
end
