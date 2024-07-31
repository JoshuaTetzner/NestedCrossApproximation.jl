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

    xhat = Vector{Vector{eltype(y)}}(
        undef, length(A.tree.trial_cluster.nodes)
    ) 
    yhat = Vector{Vector{eltype(y)}}(
        undef, length(A.tree.test_cluster.nodes)
    ) 
    
    for idx in eachindex(A.trialmomentcollection)
        if isassigned(A.trialmomentcollection, idx)
            nb = A.trialmomentcollection[idx]
            if nb.children == []
                xhat[idx] = nb.T * x[nb.σ]
            else
                xhat[idx] = nb.T[1] * xhat[nb.children[1]]
                for nchd in 2:length(nb.children)
                    xhat[idx] +=nb.T[nchd] * xhat[nb.children[nchd]]
                end
            end
        end
    end
    for idx in eachindex(A.i2itranslator)
        if isassigned(A.i2itranslator, idx)
            nb = A.i2itranslator[idx]
            if nb.children == []
                xhat[idx] = nb.T * x[nb.σ]
            else
                xhat[idx] = nb.T[1] * xhat[nb.children[1]]
                for nchd in 2:length(nb.children)
                    xhat[idx] +=nb.T[nchd] * xhat[nb.children[nchd]]
                end
            end
        end
    end

    for lrb in A.i2otranslator
        if isassigned(yhat, lrb.row_basis)
            yhat[lrb.row_basis] += lrb.Z.M * xhat[lrb.col_basis]
        else
            yhat[lrb.row_basis] = lrb.Z.M * xhat[lrb.col_basis]
        end
    end

    for idx in eachindex(A.o2otranslator)
        if isassigned(A.o2otranslator, idx)
            nb = A.o2otranslator[idx]
            if nb.children == []
                y[nb.τ] = nb.T * yhat[idx]
            else
                for chd in eachindex(nb.children)
                    if isassigned(yhat, nb.children[chd])
                        yhat[nb.children[chd]] += nb.T[chd] * yhat[idx]
                    else
                        yhat[nb.children[chd]] = nb.T[chd] * yhat[idx]
                    end
                end
            end
        end
    end
    for idx in eachindex(A.testmomentcollection)
        if isassigned(A.testmomentcollection, idx)
            nb = A.testmomentcollection[idx]
            if nb.children == []
                y[nb.τ] = nb.T * yhat[idx]
            else
                for chd in eachindex(nb.children)
                    if isassigned(yhat, nb.children[chd])
                        yhat[nb.children[chd]] += nb.T[chd] * yhat[idx]
                    else
                        yhat[nb.children[chd]] = nb.T[chd] * yhat[idx]
                    end
                end
            end
        end
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

    xhat = Vector{Vector{eltype(y)}}(
        undef, length(A.lmap.tree.test_cluster.nodes)
    ) 
    yhat = Vector{Vector{eltype(y)}}(
        undef, length(A.lmap.tree.trial_cluster.nodes)
    ) 

    for idx in eachindex(A.lmap.testmomentcollection)
        if isassigned(A.lmap.testmomentcollection, idx)
            nb = A.lmap.testmomentcollection[idx]
            xhat[idx] = transpose(nb.T) * x[nb.τ]
        end
    end

    for idx in eachindex(A.lmap.o2otranslator)
        if isassigned(A.lmap.o2otranslator, idx)
            nb = A.lmap.o2otranslator[idx]
            xhat[idx] = transpose(nb.T[1]) * xhat[nb.children[1]]
            for nchd in 2:length(nb.children)
                xhat[idx] += transpose(nb.T[nchd]) * xhat[nb.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.i2otranslator
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += transpose(lrb.Z.M) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = transpose(lrb.Z.M) * xhat[lrb.row_basis]
        end
    end

    for idx in eachindex(A.lmap.i2itranslator)
        if isassigned(A.lmap.i2itranslator, idx)
            nb = A.lmap.i2itranslator[idx]
            for chd in eachindex(nb.children)
                if isassigned(yhat, nb.children[chd])
                    yhat[nb.children[chd]] += transpose(nb.T[chd]) * yhat[idx]
                else
                    yhat[nb.children[chd]] = transpose(nb.T[chd]) * yhat[idx]
                end
            end
        end
    end

    for idx in eachindex(A.lmap.trialmomentcollection)
        if isassigned(A.lmap.trialmomentcollection, idx)
            nb = A.lmap.trialmomentcollection[idx]
            y[nb.σ] = transpose(nb.T) * yhat[idx]
        end
    end

    y += transpose(A.lmap.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, 
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinNCA}, 
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A.lmap, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Vector{eltype(y)}}(
        undef, length(A.lmap.tree.test_cluster.nodes)
    ) 
    yhat = Vector{Vector{eltype(y)}}(
        undef, length(A.lmap.tree.trial_cluster.nodes)
    ) 

    for idx in eachindex(A.lmap.testmomentcollection)
        if isassigned(A.lmap.testmomentcollection, idx)
            nb = A.lmap.testmomentcollection[idx]
            xhat[idx] = adjoint(nb.T) * x[nb.τ]
        end
    end

    for idx in eachindex(A.lmap.o2otranslator)
        if isassigned(A.lmap.o2otranslator, idx)
            nb = A.lmap.o2otranslator[idx]
            xhat[idx] = adjoint(nb.T[1]) * xhat[nb.children[1]]
            for nchd in 2:length(nb.children)
                xhat[idx] += adjoint(nb.T[nchd]) * xhat[nb.children[nchd]]
            end
        end
    end

    for lrb in A.lmap.i2otranslator
        if isassigned(yhat, lrb.col_basis)
            yhat[lrb.col_basis] += adjoint(lrb.Z.M) * xhat[lrb.row_basis]
        else
            yhat[lrb.col_basis] = adjoint(lrb.Z.M) * xhat[lrb.row_basis]
        end
    end

    for idx in eachindex(A.lmap.i2itranslator)
        if isassigned(A.lmap.i2itranslator, idx)
            nb = A.lmap.i2itranslator[idx]
            for chd in eachindex(nb.children)
                if isassigned(yhat, nb.children[chd])
                    yhat[nb.children[chd]] += adjoint(nb.T[chd]) * yhat[idx]
                else
                    yhat[nb.children[chd]] = adjoint(nb.T[chd]) * yhat[idx]
                end
            end
        end
    end

    for idx in eachindex(A.lmap.trialmomentcollection)
        if isassigned(A.lmap.trialmomentcollection, idx)
            nb = A.lmap.trialmomentcollection[idx]
            y[nb.σ] = adjoint(nb.T) * yhat[idx]
        end
    end

    y += adjoint(A.lmap.nearinteractions) * x

    return y
end