function allocate_sym_buffer(
    ::Type{K},
    compressor::TopDownCompressor{A,Nothing},
    maxrows::Int,
    maxcolumns::Int;
    maxrank=40,
    tasks=Threads.nthreads(),
) where {K,A<:FastBEAST.LRF.ACA}
    c = Channel{Matrix{K}}(tasks)
    for task in 1:tasks
        put!(c, zeros(K, maxrank, maxcolumns))
    end
    return (zeros(K, maxrows, maxrank), zeros(K, maxrows, maxrank)), c
end

function allocate_sym_buffer(
    ::Type{K},
    compressor::TopDownCompressor{CT,RT},
    maxrows::Int,
    maxcolumns::Int;
    maxrank=40,
    tasks=Threads.nthreads(),
) where {K,CT<:iACA,RT<:Nothing}
    c = Channel{Matrix{K}}(tasks)
    for task in 1:tasks
        put!(c, zeros(K, maxrank, maxrank))
    end
    return (zeros(K, maxrows, maxrank), zeros(K, maxrows, maxrank)), c
end

function allocate_sym_buffer(
    ::Type{K},
    compressor::TopDownCompressor{CT,RT},
    maxrows::Int,
    maxcolumns::Int;
    maxrank=40,
    tasks=Threads.nthreads(),
) where {K,CT<:FastBEAST.LRF.ACA,RT<:Representor}
    c = Channel{Matrix{K}}(tasks)
    for task in 1:tasks
        put!(c, zeros(K, maxrank, compressor.representor.N))
    end
    return (zeros(K, maxrows, maxrank), zeros(K, maxrows, maxrank)), c
end
