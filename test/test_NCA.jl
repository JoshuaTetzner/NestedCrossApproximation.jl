using BEAST
using NestedCrossApproximation
using CompScienceMeshes
using H2Trees
##

λ = 2
k = 2 * pi / λ
ηₗ = 1.0
ηₕ = 4.0
Γ = meshsphere(1.0, 0.05)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
tree = H2Trees.TwoNTree(space, 0.0; minvalues=200)

##

NestedCrossApproximation.AbstractKernelMatrix(op, space, space)
##
NestedCrossApproximation.PetrovGalerkinNCA2(op, space, space, tree)

NestedCrossApproximation.AbstractKernelMatrix(k, k, k)
