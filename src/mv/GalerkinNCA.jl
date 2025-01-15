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

    if A.momentcollection isa Vector
        for idx in eachindex(A.momentcollection)
            if isassigned(A.momentcollection, idx)
                nb = A.momentcollection[idx]
                xhat[idx] = transpose(nb.T) * x[nb.τ]
            end
        end
    else
        for (idx, moment) in A.momentcollection
            xhat[idx] = transpose(moment.T) * x[moment.τ]
        end
    end

    if !isassigned(A.translator, 1)
        for idx in reverse(eachindex(A.translator))
            if isassigned(A.translator, idx)
                nb = A.translator[idx]
                xhat[idx] = transpose(nb.T[1]) * xhat[nb.children[1]]
                for nchd in 2:length(nb.children)
                    xhat[idx] += transpose(nb.T[nchd]) * xhat[nb.children[nchd]]
                end
            end
        end
    else
        for level in reverse(A.translator)
            for (idx, o2o) in level
                xhat[idx] = transpose(o2o.T[1]) * xhat[o2o.children[1]]
                for nchd in 2:length(o2o.children)
                    xhat[idx] += transpose(o2o.T[nchd]) * xhat[o2o.children[nchd]]
                end
            end
        end
    end

    for lrb in A.i2otranslator
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += lrb.Z * xhat[lrb.col_basis]
        else
            holder = lrb.Z * xhat[lrb.col_basis]
            yhat[lrb.row_basis] = holder
        end
    end

    if !isassigned(A.translator, 1)
        for idx in eachindex(A.translator)
            if isassigned(A.translator, idx)
                nb = A.translator[idx]
                for chd in eachindex(nb.children)
                    if isassigned(yhat, nb.children[chd])
                        yhat[nb.children[chd]] += nb.T[chd] * yhat[idx]
                    else
                        yhat[nb.children[chd]] = nb.T[chd] * yhat[idx]
                    end
                end
            end
        end
    else
        for level in A.translator
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
    end

    if A.momentcollection isa Vector
        for idx in eachindex(A.momentcollection)
            if isassigned(A.momentcollection, idx)
                nb = A.momentcollection[idx]
                y[nb.τ] = nb.T * yhat[idx]
            end
        end
    else
        for (idx, moment) in A.momentcollection
            y[moment.τ] = moment.T * yhat[idx]
        end
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

    for idx in eachindex(A.lmap.momentcollection)
        if isassigned(A.lmap.momentcollection, idx)
            nb = A.lmap.momentcollection[idx]
            xhat[idx] = adjoint(nb.T) * x[nb.τ]
        end
    end

    for idx in reverse(eachindex(A.lmap.translator))
        if isassigned(A.lmap.translator, idx)
            nb = A.lmap.translator[idx]
            xhat[idx] = adjoint(nb.T[1]) * xhat[nb.children[1]]
            for nchd in 2:length(nb.children)
                xhat[idx] += adjoint(nb.T[nchd]) * xhat[nb.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.i2otranslator
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += conj(lrb.Z) * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = conj(lrb.Z) * xhat[lrb.col_basis]
        end
    end

    for idx in eachindex(A.lmap.translator)
        if isassigned(A.lmap.translator, idx)
            nb = A.lmap.translator[idx]
            for chd in eachindex(nb.children)
                if isassigned(yhat, nb.children[chd])
                    yhat[nb.children[chd]] += conj(nb.T[chd]) * yhat[idx]
                else
                    yhat[nb.children[chd]] = conj(nb.T[chd]) * yhat[idx]
                end
            end
        end
    end
    for idx in eachindex(A.lmap.momentcollection)
        if isassigned(A.lmap.momentcollection, idx)
            nb = A.lmap.momentcollection[idx]
            y[nb.τ] = conj(nb.T) * yhat[idx]
        end
    end

    y += adjoint(A.lmap.nearinteractions) * x

    return y
end
