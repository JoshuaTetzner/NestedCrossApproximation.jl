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
tp = translate(meshrectangle(3.0, 1.0, 0.1), SVector(3.0, 0.0, 0.0))

##
k = 10.0
SL = Maxwell3D.singlelayer(wavenumber=k)
X1 = raviartthomas(sp)
X2 = raviartthomas(tp)
T = assemble(SL, X1, X2)
println("Rank: ", rank(T))
##
@views blkasm = BEAST.blockassembler(SL, X1, X2)
 
@views function assembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata, sdata, store)
end


lm = FastBEAST.LazyMatrix(assembler, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)), ComplexF64)
##
am_rm = NestedCrossApproximation.allocate_pca_memory_rm(ComplexF64, numfunctions(X1), numfunctions(X2), maxrank=200)

fct(x) = 1/x^3 + 1/x
pivstrat = NestedCrossApproximation.PCAPivoting(
    fct, SVector(0.5,0.5,0.0), X2.pos
);

##

@time U, V, r, c = NestedCrossApproximation.pca_rm(
    lm,
    am_rm,
    pivstrat,
    tol=1e-4,
);
##
rows = zeros(Int, length(X1.pos))
cols = zeros(Int, length(X2.pos))

colbuffer = zeros(ComplexF64, length(X1.pos), 20)
cpivstrat = NestedCrossApproximation.MyPivoting(fct, X2.pos, ref=SVector(0.5, 0.5, 0.0))
rpivstrat = NestedCrossApproximation.MaximumValue()


cpivots = Int[]
for i = 1:20
    push!(cpivots, cpivstrat())
end
##
lm2 = FastBEAST.LazyMatrix(assembler, Vector(1:numfunctions(X1)), cpivots, ComplexF64)


@time r2 = NestedCrossApproximation.pca(
    lm, rows, rowbuffer, colbuffer, rpivstrat; tol=1e-4);
##
r
r2
##
@views farblkassembler = BEAST.blockassembler(
    SL, X1, X2
)
@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkassembler(tdata,sdata,store)
end

lm = FastBEAST.LazyMatrix(
    farassembler, Vector(1:numfunctions(X1)), Vector(1:numfunctions(X2)), ComplexF64
)

##
X = zeros(ComplexF64, numfunctions(X1),numfunctions(X2))

@time for i = 1:100
    lm.μ(X, Vector(1:numfunctions(X1)), Vector(1:i))
end

XX = zeros(ComplexF64, numfunctions(X1),numfunctions(X2))
@time for i = 1:100
    farassembler(XX, Vector(1:numfunctions(X1)), Vector(1:i))
end

#
##
ClusterTrees.leaves(tree)