using NestedCrossApproximation
using ParallelKMeans
using H2Trees
using CompScienceMeshes
using Test
using Random
using LinearAlgebra
using StaticArrays

points = [@SVector rand(3) for _ in 1:1000]
numfcts = length(points)

isnear = NestedCrossApproximation.isnear()

# KMeansTree

ttree = KMeansTree(points, 2; minvalues=10)
stree = KMeansTree(points, 2; minvalues=10)

tree = BlockTree(ttree, stree)
values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
_, _, fars = NestedCrossApproximation.farinteractions(tree; isnear=isnear)

A = zeros(Bool, numfcts, numfcts)
for i in eachindex(values)
    @test !any(A[values[i], nearvalues[i]])
    A[values[i], nearvalues[i]] .= true
end

for (t, s) in fars
    tvals = H2Trees.values(ttree, t)
    svals = H2Trees.values(stree, s)

    @test !(isnear(ttree, stree, t, s))
    @test !any(A[tvals, svals])
    A[tvals, svals] .= true
end

@test all(A)

# TwoNTree

tree = TwoNTree(points, points, 1 / 2^20; minvaluestest=10, minvaluestrial=10)
values, nearvalues = NestedCrossApproximation.nearinteractions(tree; isnear=isnear)
_, _, fars = NestedCrossApproximation.farinteractions(tree; isnear=isnear)
fars
A = zeros(Bool, numfcts, numfcts)
for i in eachindex(values)
    @test !any(A[values[i], nearvalues[i]])
    A[values[i], nearvalues[i]] .= true
end

for (t, s) in fars
    tvals = H2Trees.values(H2Trees.testtree(tree), t)
    svals = H2Trees.values(H2Trees.trialtree(tree), s)

    @test !(isnear(H2Trees.testtree(tree), H2Trees.trialtree(tree), t, s))
    @test !any(A[tvals, svals])
    A[tvals, svals] .= true
end

@test all(A)
