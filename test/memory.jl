struct holdmymatrix
    mat::Matrix{Float64}
end


A = zeros(100, 100)

function fill(A::Matrix{Float64})
    return A
end

function fillS(A::Matrix{Float64})
    return holdmymatrix(A)
end

@time C = fill(A);
@time B = fillS(A);

A[2, 2] = 100

B.mat[2,2]
##
using BEAST
using StaticArrays
using CompScienceMeshes

k = 10.0
Γ = meshsphere(1.0, 0.1)
space = raviartthomas(Γ)
op = Maxwell3D.singlelayer(wavenumber = k)
##
@views farblkassembler = BEAST.blockassembler(
    op, space, space
)
farblkassembler
##
@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkassembler(tdata,sdata,store)
end
##
ZS = @MMatrix zeros(ComplexF64, 100, 100) 
Z = zeros(ComplexF64, 100, 100)

@time farassembler(Z, Vector(1:100), Vector(1:100));
@time farassembler(ZS, Vector(1:100), Vector(1:100));
@time ZS = SMatrix(ZS);
##
@time Z * Z;
@time ZS * ZS;