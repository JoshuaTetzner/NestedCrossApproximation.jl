using BEAST
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using ClusterTrees
using LinearAlgebra
using Dates

filename = pwd() * "/experiments/resultstimeT.txt"
results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
fileX = open(filename, "r")
oldresults = read(fileX, String)
close(fileX)
fileX = open(filename, "w")
results = "N\t λ \t timeh2 \t timeh \t storh2 \t storh \n"
write(fileX, oldresults * "\n" * results)
close(fileX)

function storage(h2mat::NestedCrossApproximation.WidebandNCA) where {K}
    ref = size(h2mat, 1) * size(h2mat, 2)
    h2stor = 0
    for frb in h2mat.nearinteractions.blocks
        h2stor += length(frb.rowindices) * length(frb.colindices)
    end
    println("FBstorage:, ", h2stor * 8 * 10^-9)
    for lrb in h2mat.lowfrequencyinteractions
        h2stor += length(lrb.M.U) + length(lrb.M.V)
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

for h in [0.25, 0.1735, 0.122, 0.0875, 0.06, 0.0425, 0.03, 0.021]#[0.07, 0.05, 0.035, 0.024, 0.017, 0.012]#
    λ = h * 40
    k = 2 * pi / λ
    run(`gmsh experiments/typhoon.geo -2 -clmax $h  -format
      msh2 -o experiments/typhoon.msh`)
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd() * "/experiments/typhoon.msh")
    op = Maxwell3D.singlelayer(; wavenumber=k)
    space = raviartthomas(Γ)
    println("N = ", length(space.pos))

    tree = create_tree(space.pos, BoxTreeOptions(; nmin=200))
    th2 = @elapsed wnca = NestedCrossApproximation.WidebandNCA(
        op, space, space; testtree=tree, trialtree=tree, ηₕ=5.0, maxrank=50, tol=1e-3
    )
    th = @elapsed ref = HM.assemble(
        op,
        space,
        space;
        testtree=tree,
        trialtree=tree,
        compressor=FastBEAST.ACAOptions(; tol=1e-3),
    )
    sh2 = storage(wnca)#reldiff = estimate_reldifference(wnca, ref; tol=tol * 1e-1)
    sh = FastBEAST.HM.storage(ref)
    #reldiff = estimate_reldifference(wnca, ref; tol=tol)

    file = open(filename, "r")
    oldresults = read(file, String)
    close(file)
    file = open(filename, "w")
    results =
        oldresults *
        string(length(space.pos)) *
        "\t" *
        string(λ) *
        "\t" *
        string(th2) *
        "\t" *
        string(th) *
        "\t" *
        string(sh2) *
        "\t" *
        string(sh) *
        "\n"
    write(file, results)
    close(file)
    #--------------------
end
