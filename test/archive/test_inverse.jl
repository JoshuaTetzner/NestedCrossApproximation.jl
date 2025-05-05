using BEAST
using CompScienceMeshes
using FastBEAST
using LinearAlgebra
using StaticArrays

λ = 10
k = 2 * pi / λ

Γa = meshrectangle(1.0, 1.0, 0.08)
Γb = CompScienceMeshes.translate(meshrectangle(4.0, 1.0, 0.08), SVector(2.0, 0.0, 0.0))

op = Maxwell3D.singlelayer(; wavenumber=k)
tt = raviartthomas(Γa)
ts = raviartthomas(Γb)
A = assemble(op, tt, ts)

##
@views function fct(B, x, y)
    return B[:, :] = A[x, y]
end

lm = LRF.LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), ComplexF64)

r, c, U, V = LRF.aca(lm; tol=1e-8)
##
T = U * V[:, c]
norm(A[:, c] - T) / norm(A)
##
app = A[:, c] * A[r, c]^-1 * A[r, :]
##
@time b1 = A[r, c] \ A[r, :];
@time b2 = A[r, c]^-1 * A[r, :];

##
app1 = A[:, c] * b1
app2 = A[:, c] * b2
##
norm(A - app1) / norm(A)
norm(A - app2) / norm(A)
