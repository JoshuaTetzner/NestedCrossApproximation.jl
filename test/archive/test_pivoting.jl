using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation
using LinearAlgebra
using StaticArrays

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
    A_h2 = zeros(Float64, size(h2mat, 1), size(h2mat, 2))
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

function oneoverR(x, y)
    if x == y
        return 0
    else
        return 1/norm(x-y)
    end
end


Γ1 = meshrectangle(1.0, 1.0, 0.1)
Γ2 = translate(meshrectangle(1.0, 1.0, 0.1), SVector(3.5, 0.0, 0.0))

op = KernelFunction{Float64}(oneoverR)#Helmholtz3D.singlelayer()
cxd01 = FastBEAST.PointSpace{Float64}(Γ1.vertices);
cxd02 = FastBEAST.PointSpace{Float64}(Γ2.vertices);


#A = assemble(op, cxd01, cxd02)
A = zeros(Float64, length(cxd01.pos), length(cxd02.pos))
for i in eachindex(Γ1.vertices)
    for j in eachindex(Γ2.vertices)
        A[i, j] = oneoverR(Γ1.vertices[i], Γ2.vertices[j])
    end
end
##
tree1 = create_tree(cxd01.pos, KMeansTreeOptions(nmin=100, maxlevel=20))
tree2 = create_tree(cxd02.pos, KMeansTreeOptions(nmin=100, maxlevel=20))




blktree = ClusterTrees.BlockTrees.BlockTree(tree1, tree2)
nears, fars = computeinteractions(blktree, η=1.0)
##
@time h2mat = NestedCrossApproximation.PetrovGalerkinNCA(
    op, cxd01,cxd02,testtree=tree1, trialtree=tree2, compressor=FastBEAST.ACAOptions(tol=1e-5)
);


h2mat.fars#i2otranslator
##
fM = fullmat(h2mat)
println(norm(fM-A)/norm(A))


h2mat.i2otranslator[1].Z.σ
cb = A[:, h2mat.i2otranslator[1].Z.σ]
rb = A[h2mat.i2otranslator[1].Z.τ, :]
coup = A[h2mat.i2otranslator[1].Z.τ, h2mat.i2otranslator[1].Z.σ]^-1
println(norm(cb*coup*rb-A)/norm(A))
