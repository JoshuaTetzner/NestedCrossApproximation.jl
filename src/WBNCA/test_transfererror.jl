using BEAST
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using ClusterTrees
using LinearAlgebra

for h in [0.1, 0.05, 0.025, 0.0125, 0.00875]
    λ = h * 10
    k = 2 * pi / λ
    ηₗ = 1.0
    ηₕ = 5.0
    Γ = meshsphere(1.0, h)
    op = Maxwell3D.singlelayer(; wavenumber=k)
    space = raviartthomas(Γ)
    println("N = ", length(space.pos))
    tree = create_tree(space.pos, BoxTreeOptions(; nmin=100))

    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, hffars, lffars = NestedCrossApproximation.computeinteractionshf(
        blktree, k; ηₗ=ηₗ, ηₕ=ηₕ
    )

    @time wnca = NestedCrossApproximation.WidebandNCA(
        op, space, space; testtree=tree, trialtree=tree, multithreading=true, tol=1e-3
    )
    ref = HM.assemble(
        op,
        space,
        space;
        testtree=tree,
        trialtree=tree,
        compressor=FastBEAST.ACAOptions(; tol=1e-6),
    )

    reldiff = estimate_reldifference(wnca, ref)
    println(reldiff)
end
##

A = assemble(op, space, space)
##
x = rand(ComplexF64, size(A, 2))

norm(A * x - wnca * x) / norm(A * x)
norm(transpose(A) * x - transpose(wnca) * x) / norm(transpose(A) * x)
norm(adjoint(A) * x - adjoint(wnca) * x) / norm(adjoint(A) * x)
##
@time A * x;
@time wnca * x;
##

Ah2, Are = fullmat2(wnca, A)

norm(Ah2 - Are) / norm(Are)

##
function storage(h2mat::NestedCrossApproximation.WidebandNCA{K}) where {K}
    ref = size(h2mat, 1) * size(h2mat, 2)
    h2stor = 0
    for frb in h2mat.nearinteractions.blocks
        h2stor += length(frb.rowindices) * length(frb.colindices)
    end
    println("FBstorage:, ", h2stor * 8 * 10^-9)
    for lrb in h2mat.lowfrequencyinteractions
        h2stor += length(lrb.U) + length(lrb.V)
    end
    println("FB_lrbstorage:, ", h2stor * 8 * 10^-9)
    for lrb in h2mat.couplingmatrices
        h2stor += length(lrb.Z)
    end
    println("FB_lrb_couplingstorage:, ", h2stor * 8 * 10^-9)
    for (ind, dntb) in h2mat.nestedtestbases
        for (ind, ntb) in dntb
            h2stor += length(ntb)
        end
    end

    for (ind, dntb) in h2mat.nestedtrialbases
        for (ind, ntb) in dntb
            h2stor += length(ntb)
        end
    end

    for level in h2mat.testtransfermatrices
        for (ind, dtran) in level
            for (ind, tran) in dtran
                for t in tran
                    h2stor += length(t)
                end
            end
        end
    end
    for level in h2mat.trialtransfermatrices
        for (ind, dtran) in level
            for (ind, tran) in dtran
                for t in tran
                    h2stor += length(t)
                end
            end
        end
    end

    return h2stor * 8 * 10^-9, h2stor / ref
end
##
storage(wnca)
length(space.pos)^2 * 8 * 10^-9
