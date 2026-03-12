using LinearMaps

function Base.size(A::GalerkinNCA, dim=nothing)
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

function Base.size(A::Adjoint{T}, dim=nothing) where {T<:GalerkinNCA}
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

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::GalerkinNCA, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, moment) in A.nestedbases
        xhat[idx] = transpose(moment.T) * x[moment.τ]
    end

    for level in reverse(A.transfermatrices)
        for (idx, o2o) in level
            xhat[idx] = transpose(o2o.T[1]) * xhat[o2o.children[1]]
            for nchd in 2:length(o2o.children)
                xhat[idx] += transpose(o2o.T[nchd]) * xhat[o2o.children[nchd]]
            end
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += lrb.Z * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = lrb.Z * xhat[lrb.col_basis]
        end
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += transpose(lrb.Z) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = transpose(lrb.Z) * xhat[lrb.row_basis]
        end
    end

    for level in A.transfermatrices
        for (idx, i2i) in level
            for chd in eachindex(i2i.children)
                if isassigned(yhat, i2i.children[chd])
                    yhat[i2i.children[chd]] += i2i.T[chd] * yhat[idx]
                else
                    yhat[i2i.children[chd]] = i2i.T[chd] * yhat[idx]
                end
            end
        end
    end

    for (idx, moment) in A.nestedbases
        y[moment.τ] = moment.T * yhat[idx]
    end

    y += A.nearinteractions * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::LinearMaps.TransposeMap{<:Any,<:GalerkinNCA}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(eltype(y)))
    return mul!(y, A.lmap, x)
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::LinearMaps.AdjointMap{<:Any,<:GalerkinNCA}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.trial_cluster.nodes))
    yhat = Vector{Vector{eltype(y)}}(undef, length(A.lmap.tree.test_cluster.nodes))

    for (idx, moment) in A.lmap.nestedbases
        xhat[idx] = adjoint(moment.T) * x[moment.τ]
    end

    for level in reverse(A.lmap.transfermatrices)
        for (idx, o2o) in level
            xhat[idx] = adjoint(o2o.T[1]) * xhat[o2o.children[1]]
            for nchd in 2:length(o2o.children)
                xhat[idx] += adjoint(o2o.T[nchd]) * xhat[o2o.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += conj(lrb.Z) * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = conj(lrb.Z) * xhat[lrb.col_basis]
        end
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += adjoint(lrb.Z) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = adjoint(lrb.Z) * xhat[lrb.row_basis]
        end
    end

    for level in A.lmap.transfermatrices
        for (idx, i2i) in level
            for chd in eachindex(i2i.children)
                if isassigned(yhat, i2i.children[chd])
                    yhat[i2i.children[chd]] += conj(i2i.T[chd]) * yhat[idx]
                else
                    yhat[i2i.children[chd]] = conj(i2i.T[chd]) * yhat[idx]
                end
            end
        end
    end

    for (idx, moment) in A.lmap.nestedbases
        y[moment.τ] = conj(moment.T) * yhat[idx]
    end

    y += adjoint(A.lmap.nearinteractions) * x

    return y
end
