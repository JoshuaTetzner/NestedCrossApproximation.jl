using NestedCrossApproximation
using StaticArrays
using Test
hs = 1.0
k = 2.0
ηₕ = 5.0

dtree = NestedCrossApproximation.directionaltree(hs, k, ηₕ);

##

ct = SVector(1.0, 0.0, 0.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-1.0, 0.0, 0.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.0, 1.0, 0.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.0, -1.0, 0.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.0, 0.0, 1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.0, 0.0, -1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 1)
@test dtree.nodes[dest].data.ℯ == ct

##
ct = SVector(1.0, 0.5, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(1.0, -0.5, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(1.0, -0.5, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(1.0, 0.5, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
##
ct = SVector(-1.0, 0.5, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-1.0, -0.5, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-1.0, -0.5, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-1.0, 0.5, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct

##
ct = SVector(0.5, 1.0, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.5, 1.0, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, 1.0, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, 1.0, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
##
ct = SVector(0.5, -1.0, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.5, -1.0, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, -1.0, -0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, -1.0, 0.5)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
##
ct = SVector(0.5, 0.5, 1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.5, -0.5, 1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, -0.5, 1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, 0.5, 1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
##
ct = SVector(0.5, 0.5, -1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(0.5, -0.5, -1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, -0.5, -1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
ct = SVector(-0.5, 0.5, -1.0)
dest = NestedCrossApproximation.find_direction(dtree, ct, 2)
@test dtree.nodes[dest].data.ℯ == ct
