using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using LinearAlgebra
using NestedCrossApproximation

function lbases(h2mat::NestedCrossApproximation.GalerkinNCA{K}) where {K}
    trialbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
    testbases = Vector{Matrix{K}}(undef, length(h2mat.tree.test_cluster.nodes))
    for (ind, b) in h2mat.momentcollection
        trialbases[ind] = transpose(b.T)
        testbases[ind] = b.T
    end

    for (i, level) in enumerate(reverse(h2mat.translator))
        for (i, t) in level
            T = vcat(testbases[t.children[1]] * t.T[1], testbases[t.children[2]] * t.T[2])
            trialbases[i] = transpose(T)
            testbases[i] = T
        end
    end
    return testbases, trialbases
end

function lrbmat(h2mat)
    A_h2 = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    testbases, trialbases = lbases(h2mat)
    for i2o in h2mat.i2otranslator
        A_h2[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ] = testbases[i2o.row_basis] * i2o.Z * trialbases[i2o.col_basis]
    end

    return A_h2
end
function lrbmat(A, h2mat)
    lrbA = zeros(eltype(h2mat), size(h2mat, 1), size(h2mat, 2))
    for i2o in h2mat.i2otranslator
        lrbA[value(h2mat.tree.test_cluster, i2o.row_basis), value(h2mat.tree.trial_cluster, i2o.col_basis)] = A[
            value(h2mat.tree.test_cluster, i2o.row_basis),
            value(h2mat.tree.trial_cluster, i2o.col_basis),
        ]
    end

    return lrbA
end

##
Γ = meshsphere(1.0, 0.1)
λ = 2
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
nqst(i) = BEAST.DoubleNumWiltonSauterQStrat(i, i, i, i, i, i, i, i)
fqst(i) = BEAST.DoubleNumQStrat(i, i)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=50))
A = assemble(op, space, space; quadstrat=fqst(5))

##

@time h2mat = NestedCrossApproximation.NCA(
    op,
    space;
    nearinteractionquadstrat=nqst(4),
    momentquadstrat=fqst(4),
    tree=tree,
    #compressor=comp,
    maxrank=100,
    tol=1e-10,
    multithreading=false,
);

##
#; quadstrat=nqstrat)
lrbA = lrbmat(A, h2mat)
lrbh2 = lrbmat(h2mat)

norm(lrbA)
norm(lrbh2)
norm(lrbA - lrbh2) / norm(lrbA)
##
testbases, trialbases = lbases(h2mat)
for i2o in h2mat.i2otranslator
    r = value(h2mat.tree.test_cluster, i2o.row_basis)
    c = value(h2mat.tree.trial_cluster, i2o.col_basis)
    println(norm(A[r, c] - lrbh2[r, c]) / norm(A[r, c]))
end
##

blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree; η=1.0)
fars
sortedfars = NestedCrossApproximation.testfars(length(tree.nodes), fars)
##
tol = 1e-10
## 76
r76 = value(tree, 34)
c76 = value(tree, sortedfars[34])
blk76 = A[r76, c76]
@views function fct76(B, x, y)
    return B[:, :] = blk76[x, y]
end
lm76 = LRF.LazyMatrix(fct76, Vector(1:size(blk76, 1)), Vector(1:size(blk76, 2)), ComplexF64)
r_76, c_76, Ac, Bc = LRF.aca(lm76; tol=tol)
blk76 = Ac * Bc#A[r76, c76]#
U = blk76[:, c_76] / (blk76[r_76, c_76])
## 94
c94 = value(tree, 193)
r94 = value(tree, sortedfars[193])
blk94 = A[r94, c94]
@views function fct94(B, x, y)
    return B[:, :] = blk94[x, y]
end
lm94 = LRF.LazyMatrix(fct94, Vector(1:size(blk94, 1)), Vector(1:size(blk94, 2)), ComplexF64)
r_94, c_94, _, _ = LRF.aca(lm94; tol=tol)
V = blk94[r_94, c_94] \ blk94[r_94, :]
##
r = value(tree, 34)
c = value(tree, 193)
S = A[r[r_76], c[c_94]]

norm(A[r, c] - U * S * V) / norm(A[r, c])
