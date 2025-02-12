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
function channel(::TopDownCompressor{CT,Nothing}, maxrc::Int; maxrank=40) where {CT<:iACA}
    return (maxrank, maxrank)
end

function buffer(::TopDownCompressor{CT,Nothing}, maxrc::Int; maxrank=40) where {CT<:iACA}
    return (maxrc, maxrank)
end

function allocate_buffer(
    ::Type{K},
    rc_channel::Tuple{Int,Int},
    rc_buffer::Tuple{Int,Int};
    tasks=Threads.nthreads(),
) where {K}
    c = Channel{Matrix{K}}(tasks)
    for task in 1:tasks
        put!(c, zeros(K, rc_channel))
    end
    return c, (zeros(K, rc_buffer), zeros(K, rc_buffer))
end
