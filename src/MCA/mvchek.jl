function mvleafcheck(
    nestedtestbases, nestedtrialbases, couplingmatrices, x::Vector{F}, ::Type{K}
) where {F,K}
    y = zeros(K, length(x))
    xhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, basis) in nestedtrialbases
        xhat[idx] = basis.T * x[basis.σ]
    end

    for lrb in couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += lrb.Z * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = lrb.Z * xhat[lrb.col_basis]
        end
    end

    for (idx, basis) in nestedtestbases
        y[basis.τ] = basis.T * yhat[idx]
    end

    return xhat, yhat, y
end

function mvcheck(
    nestedtestbases,
    nestedtrialbases,
    testtransfermatrices,
    trialtransfermatrices,
    couplingmatrices,
)
    xhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, basis) in A.nestedtrialbases
        xhat[idx] = basis.T * x[basis.σ]
    end

    for level in reverse(A.trialtransfermatrices)
        for (idx, Θ) in level
            xhat[idx] = Θ.T[1] * xhat[Θ.children[1]]
            for nchd in 2:length(Θ.children)
                xhat[idx] += Θ.T[nchd] * xhat[Θ.children[nchd]]
            end
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += lrb.Z * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = lrb.Z * xhat[lrb.col_basis]
        end
    end

    for level in A.testtransfermatrices
        for (idx, Θ) in level
            for chd in eachindex(Θ.children)
                if isassigned(yhat, Θ.children[chd])
                    yhat[Θ.children[chd]] += Θ.T[chd] * yhat[idx]
                else
                    yhat[Θ.children[chd]] = Θ.T[chd] * yhat[idx]
                end
            end
        end
    end

    for (idx, basis) in A.nestedtestbases
        y[basis.τ] = basis.T * yhat[idx]
    end
end
