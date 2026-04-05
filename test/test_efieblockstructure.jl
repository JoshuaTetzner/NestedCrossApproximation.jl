using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using StaticArrays

##

m = meshrectangle(1.0, 1.0, 0.05)
m2 = CompScienceMeshes.translate(
    CompScienceMeshes.rotate(m, SVector(0.0, 0.0, pi / 2)), SVector(0.0, 0.0, 3)
)

##

s1 = raviartthomas(m)
s2 = raviartthomas(m2)

idcs = Int[]
for i in eachindex(s1.fns)
    id1 = s1.fns[i][1].cellid
    id2 = s1.fns[i][2].cellid
    if isapprox(getindex(s1.pos[id1], 2), getindex(s1.pos[id2], 2); atol=0.0001)
        push!(idcs, i)
    end
end

lots = idcs
rest = setdiff(1:length(s1), idcs)

##
op = Maxwell3D.singlelayer(; wavenumber=2 * pi / 0.5)
A = assemble(op, s1, s2)

Alot = A[lots, :]
norm(Alot)
Arest = A[rest, :]
norm(Arest)
