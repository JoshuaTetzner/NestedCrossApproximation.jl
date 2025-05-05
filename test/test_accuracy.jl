using NestedCrossApproximation
using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using Test
using Random
##
λ = 4.0
k = 2 * pi / λ
Γ = meshicosphere(40, 1.0)
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
Random.seed!(1)
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=100))
testcomp = NestedCrossApproximation.TopDownCompressor(iACA(space.pos), nothing)
trialcomp = NestedCrossApproximation.TopDownCompressor(
    NestedCrossApproximation.iACA(
        space.pos;
        rowpivoting=NestedCrossApproximation.IACAPivoting(space.pos),
        columnpivoting=FastBEAST.LRF.MaximumValue(),
    ),
    nothing,
)
A = assemble(op, space, space; quadstrat=BEAST.DoubleNumQStrat(2, 3));
##
@time tp, sp, h2mat = PetrovGalerkinNCA(
    op,
    space,
    space;
    testtree=tree,
    trialtree=tree,
    testcompressor=testcomp,
    trialcompressor=trialcomp,
    tol=1e-4,
    multithreading=true,
);
h2 = NestedCrossApproximation.fullmat(h2mat);

##

lerr = zeros(Float64, length(h2mat.fars))
i = 1
for lfars in h2mat.fars
    err = 0.0
    for (t, s) in lfars
        ti = value(tree, t)
        si = value(tree, s)
        err += norm(h2[ti, si] - A[ti, si]) / norm(A[ti, si])
    end
    lerr[i] = err / length(lfars)
    i += 1
end
lerr

##
error = zeros(Float64, length(h2mat.fars))
l = 1
for lfars in h2mat.fars
    for (tidx, sidx) in lfars
        τt, σt = tp[tidx]
        τs, σs = sp[sidx]

        blk = A[value(tree, tidx), value(tree, sidx)]
        S = A[τt, σs]
        U = A[value(tree, tidx), σt] / A[τt, σt]
        V = A[τs, σs] \ A[τs, value(tree, sidx)]
        error[l] += norm(blk - U * S * V) / norm(blk)
    end
    error[l] = error[l] / length(lfars)
    l += 1
end
error
##

truem = zeros(ComplexF64, size(h2mat, 1), size(h2mat, 2))
appm = zeros(ComplexF64, size(h2mat, 1), size(h2mat, 2))
error = zeros(Float64, length(h2mat.fars))
l = 1
for lfars in h2mat.fars
    for (tidx, sidx) in lfars
        ti = value(tree, tidx)
        si = value(tree, sidx)
        truem[ti, si] = A[ti, si]

        appm[value(tree, tidx), value(tree, sidx)] = h2[ti, si]
    end
    error[l] += norm(truem - appm) / norm(truem)
    truem .= 0
    appm .= 0
    l += 1
end
error
##

#
#
#
#
#
#
#

##
error = zeros(Float64, length(h2mat.fars))
tb, sb = NestedCrossApproximation.lbases(h2mat)
l = 1
for ltd in h2mat.testtransfermatrices
    for (idx, el) in ltd
        τt, σt = tp[idx]
        rows = value(tree, idx)
        U = A[rows, σt] / A[τt, σt]
        println(norm(U - tb[idx]))
    end
end

##
h2mat.fars[9]
idx = 56
τt, σt = tp[idx]
rows = value(tree, idx)
U = A[rows, σt] / A[τt, σt]
tb[56]

chd = collect(children(tree, idx))

T1 = A[tp[chd[1]][1], σt] / A[τt, σt]
norm(h2mat.testtransfermatrices[end][idx].T[1] - T1) / norm(T1)
T2 = A[tp[chd[2]][1], σt] / A[τt, σt]
norm(h2mat.testtransfermatrices[end][idx].T[2] - T2) / norm(T2)
U1 = A[value(tree, chd[1]), tp[chd[1]][2]] / A[tp[chd[1]][1], tp[chd[1]][2]]
norm(h2mat.nestedtestbases[chd[1]].T - U1) / norm(U1)
U2 = A[value(tree, chd[2]), tp[chd[2]][2]] / A[tp[chd[2]][1], tp[chd[2]][2]]
norm(h2mat.nestedtestbases[chd[2]].T - U2) / norm(U2)
norm(U[1:44, :] .- U1 * T1) / norm(U[1:44, :])

Unew = vcat(U1 * T1, U2 * T2)
## S59 30
S = A[τt, sp[30][2]]
norm(U * S - Unew * S) / norm(U * S)
