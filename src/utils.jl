function storage(h2mat)
    ref = size(h2mat, 1)*size(h2mat, 2)
    h2stor = 0.0
    for frb in h2mat.nearinteractions.self
        h2stor += length(frb.τ)*length(frb.σ)
    end
    for frb in h2mat.nearinteractions.nears
        h2stor += length(frb.τ)*length(frb.σ)
    end
    for lrb in h2mat.i2otranslator
        h2stor += size(lrb.Z.M, 1)*size(lrb.Z.M, 2)
    end
    for ind in eachindex(h2mat.momentcollection)
        if isassigned(h2mat.momentcollection, ind)
            ntb = h2mat.momentcollection[ind]
            if ntb.T isa Vector
                for t in ntb.T
                    h2stor += size(t, 1)*size(t, 2)
                end
            else
                h2stor += size(ntb.T, 1)*size(ntb.T, 2)
            end
        end
    end
    for ind in eachindex(h2mat.translator)
        if isassigned(h2mat.translator, ind)
            ntb = h2mat.translator[ind]
            if ntb.T isa Vector
                for t in ntb.T
                    h2stor += size(t, 1)*size(t, 2)
                end
            else
                h2stor += size(ntb.T, 1)*size(ntb.T, 2)
            end
        end
    end
    
    return h2stor * 8 * 10^-9, h2stor/ref
end

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