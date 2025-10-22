
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
end

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

function allocate_buttomupbuffer(
    ::Type{K},
    rc_channel::Tuple{Int,Int},
    rc_buffer::Tuple{Int,Int};
    ntasks=Threads.nthreads(),
) where {K}
    c = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(c, zeros(K, rc_channel))
    end
    return c, zeros(K, rc_buffer)
end

function allocate_aca_buffer(
    ::Type{K}, maxrows, maxcols, maxrank; ntasks=Threads.nthreads()
) where {K}
    rowbuffer = Channel{Matrix{K}}(ntasks)
    colbuffer = Channel{Matrix{K}}(ntasks)
    for _ in 1:ntasks
        put!(rowbuffer, zeros(K, maxrank, maxcols))
        put!(colbuffer, zeros(K, maxrows, maxrank))
    end
    return rowbuffer, colbuffer
end
