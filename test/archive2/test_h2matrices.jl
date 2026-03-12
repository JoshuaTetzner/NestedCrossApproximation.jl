using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using LinearAlgebra
using Test

##

Γ = meshsphere(1.0, 0.08)
op = Maxwell3D.singlelayer(; wavenumber=1.0)
space = raviartthomas(Γ)

h2mat = NestedCrossApproximation.PetrovGalerkinNCA(op, space, space)
mat = assemble(op, space, space)

##
x = rand(eltype(mat), size(mat, 2))
norm(symh2mat * x - mat * x) / norm(mat * x)
norm(h2mat * x - mat * x) / norm(mat * x)
##
@test norm(symh2mat * x - mat * x) / norm(mat * x) < 1e-3
@test norm(h2mat * x - mat * x) / norm(mat * x) < 1e-3
##
A = assemble(op, space, space)
lrbh2 = NestedCrossApproximation.lrbmat(h2mat)
Alrb = NestedCrossApproximation.lrbmat(A, h2mat)
norm(lrbh2 - Alrb) / norm(Alrb)
##
estimate_reldifference(h2mat, mat; tol=1e-3)
