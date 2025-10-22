using NestedCrossApproximation
using BEAST
using CompScienceMeshes
using StaticArrays
using FastBEAST
using ClusterTrees
using Random

Γ = meshicosphere(20, 1.0)

λ = 5
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; alpha=0.0 * im, wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(1)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = computeinteractions(blktree; η=1.0)
sortedfars = NestedCrossApproximation.testfarfield(tree, fars)

@views farblkassembler = BEAST.blockassembler(op, space, space)
@views function farassembler(Z, tdata, sdata)
    @views store(v, m, n) = (Z[m, n] += v)
    return farblkassembler(tdata, sdata, store)
end
rowidcs = value(tree, length(tree.nodes))
colidcs = value(tree, sortedfars[end])

##
ref = tree.nodes[end].node.data.ct
cts = [tree.nodes[node].node.data.ct for node in sortedfars[end]]

iaca = iACA(
    NestedCrossApproximation.IACAPivoting2(tree, space.pos),
    LRF.MaximumValue(zeros(Bool, length(rowidcs))),
    NestedCrossApproximation.IncompleteNormEstimator(Float64[], Float64(0.0)),
)
V = zeros(ComplexF64, 50, length(space.pos))
U = zeros(ComplexF64, 50, 50)
@time rpivots, cpivots = iaca(farassembler, V, U, rowidcs, 50, 1e-4, ref, sortedfars[end])
##
using Plots
plotlyjs()
rpos = reshape(
    [point[i] for point in space.pos[rowidcs] for i in 1:3], (3, length(rowidcs))
)
cpos = reshape(
    [point[i] for point in space.pos[colidcs] for i in 1:3], (3, length(colidcs))
)
rpiv = reshape(
    [point[i] for point in space.pos[rpivots] for i in 1:3], (3, length(rpivots))
)
cpiv = reshape(
    [point[i] for point in space.pos[cpivots] for i in 1:3], (3, length(cpivots))
)
##
scatter(rpos[1, :], rpos[2, :], rpos[3, :])
scatter!(cpos[1, :], cpos[2, :], cpos[3, :])
scatter!(rpiv[1, :], rpiv[2, :], rpiv[3, :])
scatter!(cpiv[1, :], cpiv[2, :], cpiv[3, :])
#norm(A - (A[:, cpivots] / A[rpivots, cpivots]) * A[rpivots, :]) / norm(A)
##
lm = LRF.LazyMatrix(farassembler, rowidcs, colidcs, ComplexF64)
iaca = iACA(space.pos)
iaca = NestedCrossApproximation.init(iaca, lm; ref=tree.nodes[end].node.data.ct)
U = zeros(ComplexF64, length(rowidcs), 50)
V = zeros(ComplexF64, 50, 50)
@time r2pivots, c2pivots = iaca(lm, V, U, 100, 1e-4);
##
r2pivots = rowidcs[r2pivots]
c2pivots = colidcs[c2pivots]
rpos2 = reshape(
    [point[i] for point in space.pos[r2pivots] for i in 1:3], (3, length(r2pivots))
)
cpos2 = reshape(
    [point[i] for point in space.pos[c2pivots] for i in 1:3], (3, length(c2pivots))
)
##
scatter!(rpos2[1, :], rpos2[2, :], rpos2[3, :])
scatter!(cpos2[1, :], cpos2[2, :], cpos2[3, :])
