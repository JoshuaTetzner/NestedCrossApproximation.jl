using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using LinearAlgebra
using Test
using StaticArrays
##
k = 0.5
Γt = meshrectangle(1.0, 1.0, 0.05)
Γs = CompScienceMeshes.translate(meshrectangle(4.0, 1.0, 0.05), SVector(3.0, 0.0, 0.0))
#Γs2 = CompScienceMeshes.translate(meshrectangle(2.0, 2.0, 0.1), SVector(0.0, 2.0, 0.0))
#Γs = weld(Γs, Γs2)
op = Maxwell3D.singlelayer(; wavenumber=k)
t = raviartthomas(Γt)
Ft = raviartthomas(Γs)

A = assemble(op, t, Ft)

_, s, _ = svd(A)
(s ./ s[1])[1:20]
