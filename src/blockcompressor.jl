function blockcompressor(
    blkmatrix::BEASTKernelMatrix{K},
    values::Vector{T},
    farvalues::Vector{Vector{T}},
    compressor::AdaptiveCrossApproximation.ACA;
    maxrank=40,
    ntasks=Threads.nthreads(),
) where {K,T<:Vector{Int}}
    lk = Threads.SpinLock()

    blocks = MatrixBlock{Int,eltype(blkmatrix),LowRankMatrix{eltype(blkmatrix)}}[]
    values == [] && return blocks
    maxcols = maximum(length.(Iterators.flatten(farvalues)))
    maxrows = maximum(length.(values))
    rowbuffer, colbuffer = allocate_aca_buffer(eltype(blkmatrix), maxrows, maxcols, maxrank)
    @tasks for tidx in eachindex(values)
        @set ntasks = ntasks
        localrowbuffer = take!(rowbuffer)
        localcolbuffer = take!(colbuffer)
        for s in farvalues[tidx]
            npivots = compressor(
                blkmatrix,
                localcolbuffer,
                localrowbuffer,
                min(maxrank, min(length(values[tidx]), length(s)));
                rowidcs=values[tidx],
                colidcs=s,
            )
            blk = MatrixBlock{Int,K,LowRankMatrix{K}}(
                LowRankMatrix(
                    localcolbuffer[1:length(values[tidx]), 1:npivots],
                    localrowbuffer[1:npivots, 1:length(s)],
                ),
                values[tidx],
                s,
            )
            lock(lk) do
                push!(blocks, blk)
            end
            localcolbuffer[1:length(values[tidx]), 1:npivots] .= 0
            localrowbuffer[1:npivots, 1:length(s)] .= 0
        end
        put!(rowbuffer, localrowbuffer)
        put!(colbuffer, localcolbuffer)
    end

    return blocks
end
