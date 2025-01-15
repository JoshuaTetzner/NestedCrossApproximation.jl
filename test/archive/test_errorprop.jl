using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation
using LinearAlgebra

function fulltestblock(h2mat, idx::I) where {I}
    if !ClusterTrees.haschildren(h2mat.tree.test_cluster, idx)
        return h2mat.testmomentcollection[idx].T
    else
        fblk = fulltestblock(h2mat, h2mat.o2otranslator[idx].children[1]) * 
            h2mat.o2otranslator[idx].T[1]
        for childidx in 2:length(h2mat.o2otranslator[idx].children)
            fblk = vcat(
                fblk,
                fulltestblock(h2mat, h2mat.o2otranslator[idx].children[childidx]) * 
                    h2mat.o2otranslator[idx].T[childidx],
            )
        end

        return fblk
    end
end

function fulltrialblock(h2mat, idx::I) where {I}
    if !ClusterTrees.haschildren(h2mat.tree.trial_cluster, idx)
        return  h2mat.trialmomentcollection[idx].T
    else
        fblk =  h2mat.i2itranslator[idx].T[1] * fulltrialblock(
            h2mat,  h2mat.i2itranslator[idx].children[1]
        )
        for childidx in 2:length(h2mat.i2itranslator[idx].children)
            fblk = hcat(
                fblk,
                h2mat.i2itranslator[idx].T[childidx] * fulltrialblock(
                    h2mat,  h2mat.i2itranslator[idx].children[childidx]
                )
            )
        end

        return fblk
    end
end

function fullmat(h2mat)
    A_h2 = zeros(ComplexF64, size(A, 1), size(A, 2))
    for M in h2mat.nearinteractions.blocks
        A_h2[M.rowindices, M.colindices] = M.matrix
    end

    for i2o in h2mat.i2otranslator
        A_h2[value(h2mat.tree.test_cluster, i2o.row_basis), value(h2mat.tree.trial_cluster, i2o.col_basis)] = 
        fulltestblock(h2mat, i2o.row_basis) * i2o.Z * fulltrialblock(
            h2mat, i2o.col_basis
        )
    end
    return A_h2
end
##
λ = 4.0
k = 2 * π / λ

Γ = meshsphere(1.0, 0.04)

op = Maxwell3D.singlelayer(wavenumber=k)
space = raviartthomas(Γ);

A = assemble(op, space, space)
##
tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, maxlevel=20))
##
function fct(r::F) where F <: Real
    return abs((1-k^2)/r + 2/r^3 - 2*im *k/r^2)
end
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    NestedCrossApproximation.PCAPivoting(fct, space.pos),
    maxrank=50,
    tol=10^-4
);
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op, space,space, compressor=compressor#FastBEAST.ACAOptions(tol=1e-4)
);
#@time hmat = HM.assemble(op, space, space, compressor=FastBEAST.ACAOptions(tol=1e-6))
##
#estimate_reldifference(h2mat, hmat)
##
fM = fullmat(h2mat)
norm(fM-A)/norm(A)
x = rand(size(A, 2))
norm(fM*x - A*x)/norm(A*x)
##
errs = [[] for i in h2mat.fars]
imp = []
lmat = zeros(ComplexF64, size(A, 1), size(A, 2))
for l in eachindex(h2mat.fars)
    for far in h2mat.fars[l]
        r = value(h2mat.tree.test_cluster, far[1])
        c = value(h2mat.tree.trial_cluster, far[2])
        lmat[r, c] = A[r, c]
        push!(errs[l], norm(fM[r, c]-A[r, c])/norm(A[r, c]))
    end
    push!(imp, norm(lmat)/norm(A))
end

for err in eachindex(errs)
    if errs[err] !=[]
        println(sum(errs[err])/length(errs[err]), ", Imp: ", imp[err], ", Mult: ", sum(errs[err])/length(errs[err])*imp[err])
    end
end


##
