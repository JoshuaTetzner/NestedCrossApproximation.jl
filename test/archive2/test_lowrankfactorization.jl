using NestedCrossApproximation
using StaticArrays
using FastBEAST
using Test

maxrank = 15
m = 50;
n = 200;
pos = [@SVector rand(3) for i in 1:n]
A = rand(ComplexF64, m, n)
@views function fct(B, x, y)
    return B[:, :] = A[x, y]
end
lm = LRF.LazyMatrix(fct, Vector(1:m), Vector(1:n), ComplexF64)

testidcs = (randperm(50)[1:m])[1:30]
trialidcs = (randperm(200)[1:n])[1:100]

#ACA
aca = NestedCrossApproximation.TopDownCompressor()
aca_cb = zeros(eltype(A), m, maxrank)
aca_rb = Channel{Matrix{eltype(A)}}(1)
put!(aca_rb, zeros(eltype(A), maxrank, n))

aca_piv = aca(aca_cb, aca_rb, fct, testidcs, trialidcs; tol=1e-4, maxrank=maxrank)
@test aca_cb[aca_piv[1], 1:maxrank] ≈ A[aca_piv[1], aca_piv[2]]
#iACA
iaca = NestedCrossApproximation.TopDownCompressor(; factorization=iACA(pos))
iaca_cb = zeros(eltype(A), m, maxrank)
iaca_rb = Channel{Matrix{eltype(A)}}(1)
put!(iaca_rb, zeros(eltype(A), maxrank, maxrank))

iaca_piv = iaca(iaca_cb, iaca_rb, fct, testidcs, trialidcs; tol=1e-4, maxrank=maxrank)
@test iaca_cb[iaca_piv[1], 1:maxrank] ≈ A[iaca_piv[1], iaca_piv[2]]

#Chebyshev
chebaca = NestedCrossApproximation.TopDownCompressor(;
    representor=ChebyshevRep(1e-3, 1.0, pos; N=3)
)
cheb_cb = zeros(eltype(A), m, maxrank)
cheb_rb = Channel{Matrix{eltype(A)}}(1)
put!(cheb_rb, zeros(eltype(A), maxrank, n))

cheb_piv = chebaca(cheb_cb, cheb_rb, fct, testidcs, trialidcs; tol=1e-4, maxrank=maxrank)
@test cheb_cb[cheb_piv[1], 1:maxrank] ≈ A[cheb_piv[1], cheb_piv[2]]
