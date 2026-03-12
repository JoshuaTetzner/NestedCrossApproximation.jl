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
    xhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(trialtree(A.tree)))
    yhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(testtree(A.tree)))

    @tasks for (idx, basis) in collect(A.nestedtrialbases)
        @set ntasks = A.ntasks
        xhat[idx] = basis * x[H2Trees.values(trialtree(A.tree), idx)]
    end

    for level in reverse(levels(trialtree(A.tree)))
        @tasks for s in collect(LevelIterator(trialtree(A.tree), level))
            @set ntasks = A.ntasks

            if haskey(A.trialtransfermatrices, s)
                chds = collect(H2Trees.children(trialtree(A.tree), s))
                xhat[s] = mapreduce(+, enumerate(chds)) do (nchd, child)
                    A.trialtransfermatrices[s][nchd] * xhat[child]
                end
            end
        end
    end

    @tasks for (t, cmats) in collect(A.couplingmatrices)
        @set ntasks = A.ntasks
        yhat[t] = mapreduce(+, cmats) do (s, cmat)
            cmat * xhat[s]
        end
    end

    for level in levels(testtree(A.tree))
        @tasks for t in collect(LevelIterator(testtree(A.tree), level))
            @set ntasks = A.ntasks
            if haskey(A.testtransfermatrices, t)
                chds = collect(H2Trees.children(testtree(A.tree), t))
                for (idx, chd) in enumerate(chds)
                    if isassigned(yhat, chd)
                        yhat[chd] += A.testtransfermatrices[t][idx] * yhat[t]
                    else
                        yhat[chd] = A.testtransfermatrices[t][idx] * yhat[t]
                    end
                end
            end
        end
    end

    OhMyThreads.@allow_boxed_captures @tasks for (idx, basis) in collect(A.nestedtestbases)
        @set ntasks = A.ntasks
        y[H2Trees.values(testtree(A.tree), idx)] = basis * yhat[idx]
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

    xhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(testtree(A.lmap.tree)))
    yhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(trialtree(A.lmap.tree)))

    @tasks for (idx, basis) in collect(A.lmap.nestedtestbases)
        @set ntasks = A.lmap.ntasks
        xhat[idx] = transpose(basis) * x[H2Trees.values(testtree(A.lmap.tree), idx)]
    end

    for level in reverse(levels(testtree(A.lmap.tree)))
        @tasks for t in collect(LevelIterator(testtree(A.lmap.tree), level))
            @set ntasks = A.lmap.ntasks
            if haskey(A.lmap.testtransfermatrices, t)
                chds = collect(H2Trees.children(testtree(A.lmap.tree), t))
                xhat[t] = mapreduce(+, enumerate(chds)) do (nchd, child)
                    transpose(A.lmap.testtransfermatrices[t][nchd]) * xhat[child]
                end
            end
        end
    end

    ## no easy multithreading here
    for (t, cmats) in collect(A.lmap.couplingmatrices)
        for (s, cmat) in cmats
            if isassigned(yhat, s)
                yhat[s] += transpose(cmat) * xhat[t]
            else
                yhat[s] = transpose(cmat) * xhat[t]
            end
        end
    end

    for level in levels(trialtree(A.lmap.tree))
        @tasks for s in collect(LevelIterator(trialtree(A.lmap.tree), level))
            @set ntasks = A.lmap.ntasks
            if haskey(A.lmap.trialtransfermatrices, s)
                chds = collect(H2Trees.children(trialtree(A.lmap.tree), s))
                for (idx, chd) in enumerate(chds)
                    if isassigned(yhat, chd)
                        yhat[chd] +=
                            transpose(A.lmap.trialtransfermatrices[s][idx]) * yhat[s]
                    else
                        yhat[chd] =
                            transpose(A.lmap.trialtransfermatrices[s][idx]) * yhat[s]
                    end
                end
            end
        end
    end

    OhMyThreads.@allow_boxed_captures @tasks for (idx, basis) in
                                                 collect(A.lmap.nestedtrialbases)
        @set ntasks = A.lmap.ntasks
        y[H2Trees.values(trialtree(A.lmap.tree), idx)] = transpose(basis) * yhat[idx]
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

    xhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(testtree(A.lmap.tree)))
    yhat = Vector{Vector{eltype(y)}}(undef, numberofnodes(trialtree(A.lmap.tree)))

    @tasks for (idx, basis) in collect(A.lmap.nestedtestbases)
        @set ntasks = A.lmap.ntasks
        xhat[idx] = adjoint(basis) * x[H2Trees.values(testtree(A.lmap.tree), idx)]
    end

    for level in reverse(levels(testtree(A.lmap.tree)))
        @tasks for t in collect(LevelIterator(testtree(A.lmap.tree), level))
            @set ntasks = A.lmap.ntasks
            if haskey(A.lmap.testtransfermatrices, t)
                chds = collect(H2Trees.children(testtree(A.lmap.tree), t))
                xhat[t] = mapreduce(+, enumerate(chds)) do (nchd, child)
                    adjoint(A.lmap.testtransfermatrices[t][nchd]) * xhat[child]
                end
            end
        end
    end

    ## no easy multithreading here
    for (t, cmats) in collect(A.lmap.couplingmatrices)
        for (s, cmat) in cmats
            if isassigned(yhat, s)
                yhat[s] += adjoint(cmat) * xhat[t]
            else
                yhat[s] = adjoint(cmat) * xhat[t]
            end
        end
    end

    for level in levels(trialtree(A.lmap.tree))
        @tasks for s in collect(LevelIterator(trialtree(A.lmap.tree), level))
            @set ntasks = A.lmap.ntasks
            if haskey(A.lmap.trialtransfermatrices, s)
                chds = collect(H2Trees.children(trialtree(A.lmap.tree), s))
                for (idx, chd) in enumerate(chds)
                    if isassigned(yhat, chd)
                        yhat[chd] += adjoint(A.lmap.trialtransfermatrices[s][idx]) * yhat[s]
                    else
                        yhat[chd] = adjoint(A.lmap.trialtransfermatrices[s][idx]) * yhat[s]
                    end
                end
            end
        end
    end

    OhMyThreads.@allow_boxed_captures @tasks for (idx, basis) in
                                                 collect(A.lmap.nestedtrialbases)
        @set ntasks = A.lmap.ntasks
        y[H2Trees.values(trialtree(A.lmap.tree), idx)] = adjoint(basis) * yhat[idx]
    end

    y += adjoint(A.lmap.nearinteractions) * x

    return y
end
