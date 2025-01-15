using BEAST
using StaticArrays
using CompScienceMeshes
using NestedCrossApproximation
using LinearAlgebra
using FastBEAST


function StoM(vec)
    mat = zeros(Float64, length(vec), 3)
    for idx in eachindex(vec)
        mat[idx, 1] = vec[idx][1]
        mat[idx, 2] = vec[idx][2]
        mat[idx, 3] = vec[idx][3]
    end
    return mat
end
##
sp = meshrectangle(1.0, 1.0, 0.1)
tp = translate(meshrectangle(1.0, 1.0, 0.1), SVector(3.0, 0.0, 0.0))
sv = StoM(sp.vertices)
tv = StoM(tp.vertices)

##
k = 10.0
SL = Maxwell3D.singlelayer(wavenumber=k)
X1 = raviartthomas(sp)
X2 = raviartthomas(tp)
sv = StoM(X1.pos)
tv = StoM(X2.pos)
T = assemble(SL, X1, X2)
println("Rank: ", rank(T))
##
@views blkasm = BEAST.blockassembler(SL, X1, X2)
 
@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata, sdata, store)
end

##
function fct(val::F) where F <: Real
    return F(1)/val
end
lm = FastBEAST.LazyMatrix(assembler, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)), ComplexF64)
am_rm = NestedCrossApproximation.allocate_pca_memory_rm(ComplexF64, numfunctions(X1), numfunctions(X2), maxrank=200)

pivstrat = NestedCrossApproximation.PCAPivoting(
    fct, SVector(0.5,0.5,0.0), X2.pos
)
@time U, V, r, c = NestedCrossApproximation.pca_rm(
    lm,
    am_rm,
    pivstrat,
    tol=1e-4,
);
length(r)
##

norm(M[])