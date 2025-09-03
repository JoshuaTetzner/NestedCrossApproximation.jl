#=
# Standard NCA
function channel(
    ::TopDownCompressor{CT,Nothing}, maxrc::Int; maxrank=40
) where {CT<:FastBEAST.LRF.ACA}
    return (maxrank, maxrc)
end

function buffer(
    ::TopDownCompressor{CT,Nothing}, maxrc::Int; maxrank=40
) where {CT<:FastBEAST.LRF.ACA}
    return (maxrc, maxrank)
end

# Bebendorf NCA
function channel(
    comp::TopDownCompressor{CT,RT}, maxrc::Int; maxrank=40
) where {CT<:FastBEAST.LRF.ACA,RT<:Representor}
    return (maxrank, comp.representor.N)
end

function buffer(
    ::TopDownCompressor{CT,RT}, maxrc::Int; maxrank=40
) where {CT<:FastBEAST.LRF.ACA,RT<:Representor}
    return (maxrc, maxrank)
end

# iACA
function channel(
    ::Union{TopDownCompressor{CT,Nothing},ButtomUpCompressor{CT,Nothing}},
    maxrc::Int;
    maxrank=40,
) where {CT<:iACA}
    return (maxrank, maxrank)
end

function buffer(
    ::Union{TopDownCompressor{CT,Nothing},ButtomUpCompressor{CT,Nothing}},
    maxrc::Int;
    maxrank=40,
) where {CT<:iACA}
    return (maxrc, maxrank)
end

function testbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_buffer(
        T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); tasks=tasks
    )
end

function trialbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return reverse(
        allocate_buffer(T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); tasks=tasks)
    )
end
=#
function testbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.ACA}
    return allocate_buffer(
        T, (maxrank, size(farmatrix, 2)), (size(farmatrix, 1), maxrank); ntasks=ntasks
    )
end

function testbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_buffer(
        T, (maxrank, maxrank), (size(farmatrix, 1), maxrank); ntasks=ntasks
    )
end

function trialbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.ACA}
    return allocate_buffer(
        T, (size(farmatrix, 1), maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
    )
end

function trialbuffer(
    ::TopDownCompressor{CT,Nothing},
    farmatrix::AbstractKernelMatrix{T};
    maxrank=50,
    ntasks=Threads.nthreads(),
) where {T,CT<:AdaptiveCrossApproximation.iACA}
    return allocate_buffer(
        T, (maxrank, maxrank), (maxrank, size(farmatrix, 2)); ntasks=ntasks
    )
end
#=
function trialbuffer(
    ::Type{K},
    rc_channel::Tuple{Int,Int},
    rc_buffer::Tuple{Int,Int};
    ntasks=Threads.nthreads(),
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, rc_channel))
    end
    return c, (zeros(K, rc_buffer), zeros(K, rc_buffer))
end=#

bufferidx(level::Int) = (iseven(level) ? (return 1) : (return 2))

function allocate_buffer(
    ::Type{K}, channel::Tuple{Int,Int}, matrix::Tuple{Int,Int}; ntasks=Threads.nthreads()
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, channel))
    end
    return c, (zeros(K, matrix), zeros(K, matrix))
end

#=
function allocate_buttomupbuffer(
    ::Type{K},
    rc_channel::Tuple{Int,Int},
    rc_buffer::Tuple{Int,Int};
    ntasks=Threads.nthreads(),
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, channel))
    end
    return c, zeros(K, buffer)
end=#
