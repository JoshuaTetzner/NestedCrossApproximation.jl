using BEAST
using FastBEAST
using CompScienceMeshes
using NestedCrossApproximation
using ClusterTrees
using LinearAlgebra
using Dates

filename = pwd() * "/experiments/resultserrorT.txt"
results = Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "\n"
fileX = open(filename, "r")
oldresults = read(fileX, String)
close(fileX)
fileX = open(filename, "w")
results = "N\t λ \t tol \t err\n"
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

for h in [0.12, 0.085]#0.05, 0.035, 
    λ = h * 20
    k = 2 * pi / λ
    run(`gmsh experiments/typhoon.geo -2 -clmax $h  -format
     msh2 -o experiments/typhoon.msh`)
    Γ = CompScienceMeshes.read_gmsh_mesh(pwd() * "/experiments/typhoon.msh")
    op = Maxwell3D.singlelayer(; wavenumber=k)
    space = raviartthomas(Γ)
    println("N = ", length(space.pos))
    ref = assemble(op, space, space)

    for tol in [1e-2, 1e-4, 1e-6]
        tree = create_tree(space.pos, BoxTreeOptions(; nmin=200))
        wnca = NestedCrossApproximation.WidebandNCA(
            op, space, space; testtree=tree, trialtree=tree, ηₕ=5.0, maxrank=200, tol=tol
        )

        reldiff = estimate_reldifference(wnca, ref; tol=tol * 0.1)

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
            string(tol) *
            "\t" *
            string(reldiff) *
            "\n"
        write(file, results)
        close(file)
        #--------------------
    end
end
