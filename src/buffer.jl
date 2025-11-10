
function testbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.ACA}
    return allocate_tdbuffer(
        T, (maxrank, size(farmatrix, 2)), (size(farmatrix, 1), maxrank); ntasks=ntasks
    )
end

function testbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_tdbuffer(
        T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); ntasks=ntasks
    )
end

function testbuffer(
    ::BottomUpCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_bubuffer(
        T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); ntasks=ntasks
    )
end

function trialbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.ACA}
    return allocate_tdbuffer(
        T, (size(farmatrix, 1), maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
    )
end

function trialbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_tdbuffer(
        T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
    )
end

function trialbuffer(
    ::BottomUpCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_bubuffer(
        T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
    )
end

bufferidx(level::Int) = (iseven(level) ? (return 1) : (return 2))

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
