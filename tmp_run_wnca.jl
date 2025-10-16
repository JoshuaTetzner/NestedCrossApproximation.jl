using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
using AdaptiveCrossApproximation
using StaticArrays
using LinearAlgebra

λ = 0.3
k = 2 * pi / λ
Γ = meshrectangle(1.0, 1.0, 0.01)
Γ2 = translate(Γ, SVector(1.0, 0.0, 0.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
spaceX = raviartthomas(Γ)
spaceY = raviartthomas(Γ2)
ttree = H2Trees.TwoNTree(spaceX, 0.02)
stree = H2Trees.TwoNTree(spaceY, 0.02)
tree = BlockTree(ttree, stree)

isnear(k, ta, tb, na, nb) = NestedCrossApproximation.isnear(k, ta, tb, na, nb)
issnear = (ta, tb, na, nb) -> isnear(k, ta, tb, na, nb)

wnca = NestedCrossApproximation.PetrovGalerkinWNCA(
    op, spaceX, spaceY, tree; isnear=issnear, lfcompressor=AdaptiveCrossApproximation.ACA()
)

x = rand(ComplexF64, size(wnca, 2))
println("computing wnca*x")
y1 = wnca * x
println("computing reconstruct* x")
y2 = NestedCrossApproximation.reconstruct(wnca) * x
println("done")
println(norm(y1 - y2) / norm(y2))
