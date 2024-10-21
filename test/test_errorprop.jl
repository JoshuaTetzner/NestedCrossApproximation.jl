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
    A_h2 = zeros(Float64, size(A, 1), size(A, 2))
    for M in h2mat.nearinteractions.M
        A_h2[M.τ, M.σ] = M.M
    end

    for i2o in h2mat.i2otranslator
        A_h2[i2o.τ, i2o.σ] = fulltestblock(h2mat, i2o.row_basis) * i2o.Z.M * fulltrialblock(
            h2mat, i2o.col_basis
        )
    end
    return A_h2
end
##

Γ = meshrectangle(2.0, 0.2, 0.009)
Γ.faces

op = Helmholtz3D.singlelayer()
cxd0 = lagrangec0d1(Γ);

A = assemble(op, cxd0, cxd0)
##
tree = create_tree(cxd0.pos, KMeansTreeOptions(nmin=30, maxlevel=20))
##
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op, cxd0,cxd0, compressor=FastBEAST.ACAOptions(tol=1e-4)
);
##
fM = fullmat(h2mat)

errs = [[] for i in h2mat.fars]
imp = []
lmat = zeros(Float64, size(A, 1), size(A, 2))
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

clusterlink = FastBEAST.cluster_link(h2mat.tree.test_cluster)