using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using CompScienceMeshes
using StaticArrays

Γ = meshrectangle(1.0, 1.0, 0.01)
#Γ2a = translate(meshrectangle(1.0, 1.0, 0.05), SVector(6.0, 0.0, 0.0))
#Γ2b = translate(meshrectangle(1.0, 1.0, 0.05), SVector(-5.0, 0.0, 0.0))
#Γ2 = weld(Γ2a, Γ2b)
##
λ = 0.2
k = 2π / λ
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
ttree = TwoNTree(Γ1.vertices, 0.25; minvalues=400)
stree = TwoNTree(Γ2.vertices, 0.25; minvalues=400)
tree = BlockTree(ttree, stree)

H2Trees.halfsize(ttree, 2)
Γ1.vertices

testfardata = NestedCrossApproximation.directionaltestfars(tree; islf=islf, isnear=isnear)
trialfardata = NestedCrossApproximation.directionaltrialfars(tree; islf=islf, isnear=isnear)
testfardata.𝓔

NestedCrossApproximation.maxlevel(ttree, islf)
H2Trees.halfsize(ttree)
islf(ttree, 5)
2 * sqrt(3) * H2Trees.halfsize(ttree) * islf.k / (2.0^(6 - 1))# <= 1
k
2 * sqrt(3) * H2Trees.halfsize(ttree) / 2^2

sqrt(3) * 2 * H2Trees.halfsize(ttree) / 2^6
islf(2 * 0.025)

##
lambda = 2 * pi#1.0
k = 2pi / lambda
diamX = 1.0

##

2 * pi * diamX < lambda

##
#angle between (1,0,0) and (1,1,1)
ang6p = acos(1 / sqrt(3))

# for N=6*2^(n+1) points on the unit sphere
angNp(n) = ang6p / 2^n

n(N) = log(2, N / (2 * 6)) + 1
n(6)

diamX = 1.0
N = 6 * 4^log(2, (acos(1 / sqrt(3)) * k * diamX))

angle(6 * 4)
##
lambda = 2 * pi#1.0
k = 2pi / lambda
diamX = 1.0

N = 6 * 4^(log(2, acos(1 / sqrt(3)) / asin(1 / (k * diamX))))
