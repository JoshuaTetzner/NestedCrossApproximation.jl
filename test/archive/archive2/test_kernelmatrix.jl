using BEAST
using NestedCrossApproximation
using CompScienceMeshes

Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

K = NestedCrossApproximation.AbstractKernelMatrix(op, space, space)

##
length(space)
blkasm = BEAST.blockassembler(op, space, space)
##
blk = zeros(Float64, 3, 3)
K(blk, 1:3, 1:3)  # sample call to test
