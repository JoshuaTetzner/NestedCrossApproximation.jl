using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using LinearAlgebra
using Test

##

Γ = meshsphere(1.0, 0.08)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)


symh2mat = NestedCrossApproximation.GalerkinNCA(op, space)
h2mat = NestedCrossApproximation.PetrovGalerkinNCA(op, space, space)
mat = assemble(op, space, space)

##
x = rand(eltype(mat), size(mat, 2))
@test norm(symh2mat*x -mat*x)/norm(mat*x) < 1e-3
@test norm(h2mat*x -mat*x)/norm(mat*x) < 1e-3
