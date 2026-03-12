bufferidx(level::Int) = iseven(level) ? 1 : 2

function allocate_tdbuffer(
    ::Type{K}, channel::Tuple{Int,Int}, matrix::Tuple{Int,Int}; ntasks=Threads.nthreads()
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, channel))
    end
    return c, (zeros(K, matrix), zeros(K, matrix))
end

function allocate_bubuffer(
    ::Type{K}, channel::Tuple{Int,Int}, matrix::Tuple{Int,Int}; ntasks=Threads.nthreads()
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, channel))
    end
    return c, zeros(K, matrix)
end

_is_bottomup(compressor) = nameof(typeof(compressor)) == :BottomUp

function testbuffer(
    compressor,
    farmatrix::AbstractKernelMatrix{T};
    maxrank::Int=40,
    ntasks::Int=Threads.nthreads(),
) where {T}
    factorization = getfield(compressor, :factorization)
    if factorization isa AdaptiveCrossApproximation.ACA
        return allocate_tdbuffer(
            T, (maxrank, size(farmatrix, 2)), (size(farmatrix, 1), maxrank); ntasks=ntasks
        )
    elseif factorization isa AdaptiveCrossApproximation.iACA
        if _is_bottomup(compressor)
            return allocate_bubuffer(
                T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); ntasks=ntasks
            )
        end
        return allocate_tdbuffer(
            T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); ntasks=ntasks
        )
    end
    return error(
        "No test buffer allocation available for compressor type $(typeof(compressor))."
    )
end

function trialbuffer(
    compressor,
    farmatrix::AbstractKernelMatrix{T};
    maxrank::Int=40,
    ntasks::Int=Threads.nthreads(),
) where {T}
    factorization = getfield(compressor, :factorization)
    if factorization isa AdaptiveCrossApproximation.ACA
        return allocate_tdbuffer(
            T, (size(farmatrix, 1), maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
        )
    elseif factorization isa AdaptiveCrossApproximation.iACA
        if _is_bottomup(compressor)
            return allocate_bubuffer(
                T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
            )
        end
        return allocate_tdbuffer(
            T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
        )
    end
    return error(
        "No trial buffer allocation available for compressor type $(typeof(compressor))."
    )
end
