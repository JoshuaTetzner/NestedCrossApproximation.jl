using BEAST
using FastBEAST
using Base.Threads
using ThreadsX
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation
using LinearAlgebra
using ChebyshevApprox
NCAM = NestedCrossApproximation

Γ = meshsphere(1.0, 0.1)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree; η=1.0)
@views farblkassembler = BEAST.blockassembler(op, space, space)
@views function farassembler(Z, tdata, sdata)
    @views store(v, m, n) = (Z[m, n] += v)
    return farblkassembler(tdata, sdata, store)
end
##
fact = NCAM.iACA(
    space.pos; rowpivoting=LRF.MaximumValue(), columnpivoting=NCAM.IACAPivoting(space.pos)
)
comp = NCAM.TopDownCompressor(; factorization=fact);
@time h2mat = NCAM.NCA(op, space; tree=tree, compressor=comp);
@time h22 = NCAM.GalerkinNCA(op, space; tree=tree);
##
x = rand(size(h2mat, 2));
fullmat = assemble(op, space, space)
norm(h22 * x - fullmat * x) / norm(fullmat * x)
norm(fullmat * x - h2mat * x) / norm(fullmat * x)
##
lm = LRF.LazyMatrix(
    farassembler, Vector(1:length(space.pos)), Vector(1:length(space.pos)), Float64
)
fact = NCAM.iACA(
    space.pos; rowpivoting=LRF.MaximumValue(), columnpivoting=NCAM.IACAPivoting(space.pos)
)

NCAM.init(fact, lm)

comp = NCAM.TopDownCompressor(; factorization=fact);
@time NCAM.compress(tree, farassembler, fars, comp, Float64);

##
@time begin
    test_fars = row_pivot_selection(tree, tree, fars, farassembler, Float64)

    momentcollection, translator = build_test_bases(tree, test_fars, Float64)
end;

##
intc = fars[end][1]
rows = value(tree, intc[1])
cols = value(tree, intc[2])
##
@time comp = NCA.iACA(
    LRF.MaximumValue(),
    NCA.IACAPivoting(space.pos),
    NCA.IncompleteNormEstimator(Float64[], 0.0),
);
lm = LRF.LazyMatrix(farassembler, rows, cols, Float64)
ref = sum(space.pos[cols]) / length(space.pos[cols])
@time compressor = NCA.init(comp, lm; ref=ref);
rowbuffer = zeros(Float64, 40, length(cols))
colbuffer = zeros(Float64, length(rows), 40)
@time r, c = compressor(lm, rowbuffer, colbuffer, 40, 1e-5);
##
r

mat = zeros(Float64, length(rows), length(cols))
lm(mat, 1:length(rows), 1:length(cols))
##
norm(mat[:, c] * mat[r, c]^-1 * mat[r, :] - mat) / norm(mat)
##
lm = FastBEAST.LazyMatrix(farassembler, rows, cols, Float64)
am_rm = NestedCrossApproximation.allocate_pca_memory_rm(
    Float64, length(rows), length(cols); maxrank=40
)
fct(r) = 1 / r .^ 2
pivstrat = NestedCrossApproximation.PCAPivoting(fct, ref, space.pos[cols])
@time U, V, r, c = NestedCrossApproximation.pca_rm(lm, am_rm, pivstrat; tol=1e-5);
r
##

using StaticArrays

using Plots
plotlyjs()
##
m = meshrectangle(1.0, 1.0, 0.03).vertices
rep = NCA.ChebyshevRep(1e-2, 2.0, m)
rep, cn = rep(Vector(1:length(m)))

pos = reshape([point[i] for point in m for i in 1:3], (3, length(m)))
pos2 = reshape([point[i] for point in m[rep] for i in 1:3], (3, length(rep)))
pos3 = reshape([point[i] for point in cn for i in 1:3], (3, length(cn)))

scatter(pos[1, :], pos[2, :], pos[3, :])
scatter!(pos2[1, :], pos2[2, :], pos2[3, :])
scatter!(pos3[1, :], pos3[2, :], pos3[3, :])

###

dd = Dict([1, 2, 3, 4] .=> [[1, 2], [2], [3], [4]])

reverse(collect(keys(dd)))
typeof(a)
