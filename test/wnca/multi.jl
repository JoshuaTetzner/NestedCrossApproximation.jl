function multi(A::NestedCrossApproximation.PetrovGalerkinWNCA, x::AbstractVector)
    y = zeros(eltype(A), size(A, 1))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trialcluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.testcluster.nodes))

    for (s, bases) in A.nestedtrialbases
        dirbases = Vector{eltype(A)}[]
        for basis in Base.values(bases)
            push!(dirbases, basis.T * x[basis.σ])
        end
        xhat[s] = Dict(keys(bases) .=> dirbases)
    end

    for level in reverse(A.trialtransfermatrices)
        for (s, transfers) in level
            bases = Vector{eltype(y)}[]
            for (dir, transfer) in transfers
                childdir = NestedCrossApproximation.parent(A.dtree, dir)
                res = transfer.T[1] * xhat[transfer.children[1]][childdir]
                for c in 2:length(transfer.children)
                    res += transfer.T[c] * xhat[transfer.children[c]][childdir]
                end
                push!(bases, res)
            end
            xhat[s] = Dict(keys(transfers) .=> bases)
        end
    end

    for coupling in A.couplingmatrices
        t = coupling.row_basis
        s = coupling.col_basis

        if isassigned(yhat, t)
            if haskey(yhat[t], coupling.dir)
                yhat[t][coupling.dir] += coupling.Z * xhat[s][coupling.dir]
            else
                yhat[t][coupling.dir] = coupling.Z * xhat[s][coupling.dir]
            end
        else
            yhat[t] = Dict(coupling.dir => coupling.Z * xhat[s][coupling.dir])
        end
    end

    for level in A.testtransfermatrices
        for (t, transfers) in level
            for (dir, transfer) in transfers
                chiddir = NestedCrossApproximation.parent(A.dtree, dir)
                for (c, child) in enumerate(transfer.children)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], chiddir)
                            yhat[child][chiddir] += transfer.T[c] * yhat[t][dir]
                        else
                            yhat[child][chiddir] = transfer.T[c] * yhat[t][dir]
                        end
                    else
                        yhat[child] = Dict(chiddir => transfer.T[c] * yhat[t][dir])
                    end
                end
            end
        end
    end

    for (t, bases) in A.nestedtestbases
        for (dir, basis) in bases
            y[basis.τ] += basis.T * yhat[t][dir]
        end
    end

    for blk in A.lowfrequencyinteractions
        y[blk.τ] += blk.M * x[blk.σ]
    end

    y += A.nearinteractions * x

    return y
end
