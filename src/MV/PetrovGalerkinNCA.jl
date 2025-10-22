using LinearMaps

function Base.size(A::PetrovGalerkinNCA, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::Adjoint{T}, dim=nothing) where {T<:PetrovGalerkinNCA}
    if dim === nothing
        return reverse(A.dim[1], A.dim[2])
    elseif dim == 1
        return h2mat.lmap.dim[2]
    elseif dim == 2
        return h2mat.lmap.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::PetrovGalerkinNCA, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    xhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, basis) in A.nestedtrialbases
        xhat[idx] = basis.T * x[basis.σ]
    end

    for level in reverse(A.trialtransfermatrices)
        for (idx, Θ) in level
            if !isassigned(xhat, Θ.children[1])
                println(idx)
                println(Θ.children[1])
            end
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

    y += A.nearinteractions * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, A.lmap, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.test_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.trial_cluster.nodes))

    for (idx, basis) in A.lmap.nestedtestbases
        xhat[idx] = transpose(basis.T) * x[basis.τ]
    end

    for level in reverse(A.lmap.testtransfermatrices)
        for (idx, Θ) in level
            xhat[idx] = transpose(Θ.T[1]) * xhat[Θ.children[1]]
            for nchd in 2:length(Θ.children)
                xhat[idx] += transpose(Θ.T[nchd]) * xhat[Θ.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += transpose(lrb.Z) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = transpose(lrb.Z) * xhat[lrb.row_basis]
        end
    end

    for level in A.lmap.trialtransfermatrices
        for (idx, Θ) in level
            for chd in eachindex(Θ.children)
                if isassigned(yhat, Θ.children[chd])
                    yhat[Θ.children[chd]] += transpose(Θ.T[chd]) * yhat[idx]
                else
                    yhat[Θ.children[chd]] = transpose(Θ.T[chd]) * yhat[idx]
                end
            end
        end
    end

    for (idx, basis) in A.lmap.nestedtrialbases
        y[basis.σ] = transpose(basis.T) * yhat[idx]
    end

    y += transpose(A.lmap.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, A.lmap, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.test_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.trial_cluster.nodes))

    for (idx, basis) in A.lmap.nestedtestbases
        xhat[idx] = adjoint(basis.T) * x[basis.τ]
    end

    for level in reverse(A.lmap.testtransfermatrices)
        for (idx, Θ) in level
            xhat[idx] = adjoint(Θ.T[1]) * xhat[Θ.children[1]]
            for nchd in 2:length(Θ.children)
                xhat[idx] += adjoint(Θ.T[nchd]) * xhat[Θ.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += adjoint(lrb.Z) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = adjoint(lrb.Z) * xhat[lrb.row_basis]
        end
    end

    for level in A.lmap.trialtransfermatrices
        for (idx, Θ) in level
            for chd in eachindex(Θ.children)
                if isassigned(yhat, Θ.children[chd])
                    yhat[Θ.children[chd]] += adjoint(Θ.T[chd]) * yhat[idx]
                else
                    yhat[Θ.children[chd]] = adjoint(Θ.T[chd]) * yhat[idx]
                end
            end
        end
    end

    for (idx, basis) in A.lmap.nestedtrialbases
        y[basis.σ] = adjoint(basis.T) * yhat[idx]
    end

    y += adjoint(A.lmap.nearinteractions) * x

    return y
end
