mutable struct TopDownCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function TopDownCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function TopDownCompressor(;
    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
)
    return TopDownCompressor(factorization, representor)
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    t::Vector{Int},
    Ft::Vector{Int},
    colbuffer::K,
    rowchannel::Channel{K},
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.ACA,RP}
    !isnothing(compressor.representor) && (t, Ft=compressor.representor(t, Ft))
    rowbuffer = take!(rowchannel)
    rows = zeros(Int, 40)
    cols = zeros(Int, 40)

    colbuffer[t, 1:40] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, t, 1:40),
        view(rowbuffer, 1:40, Ft),
        min(40, min(length(t), length(Ft)));
        rows=rows,
        cols=cols,
        rowidcs=t,
        colidcs=Ft,
    )
    colbuffer[t, 1:npivots] =
        colbuffer[t, 1:npivots] * rowbuffer[1:npivots, cols[1:npivots]]

    rowbuffer[1:npivots, Ft] .= 0.0
    put!(rowchannel, rowbuffer)

    return rows[1:npivots], cols[1:npivots]
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    t::Vector{Int},
    Ft::Vector{Int},
    colbuffer::K,
    rowchannel::Channel{K};
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.iACA,RP}
    !isnothing(compressor.representor) && (t, Ft=compressor.representor(t, Ft))
    rowbuffer = take!(rowchannel)
    maxrank = min(maxrank, min(length(t), length(Ft)))
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    colbuffer[t, 1:maxrank] .= 0.0

    npivots, rows, cols = compressor.lrf(
        farmatrix,
        view(colbuffer, t, 1:maxrank),
        rowbuffer,
        maxrank;
        rows=rows,
        cols=cols,
        rowidcs=t,
        colidcs=Ft,
    )
    colbuffer[t, 1:npivots] = colbuffer[t, 1:npivots] * rowbuffer[1:npivots, 1:npivots]
    rowbuffer[1:npivots, 1:npivots] .= 0.0
    put!(rowchannel, rowbuffer)

    return t[rows], Ft[cols]
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Int},
    s::Vector{Int},
    colchannel::Channel{K},
    rowbuffer::K,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.ACA,RP}
    !isnothing(compressor.representor) && (s, Fs=compressor.representor(s, Fs))
    colbuffer = take!(colchannel)
    rows = zeros(Int, 40)
    cols = zeros(Int, 40)

    rowbuffer[1:40, s] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, Fs, 1:40),
        view(rowbuffer, 1:40, s),
        min(40, min(length(s), length(Fs)));
        rows=rows,
        cols=cols,
        rowidcs=Fs,
        colidcs=s,
    )
    rowbuffer[1:npivots, s] =
        colbuffer[rows[1:npivots], 1:npivots] * rowbuffer[1:npivots, s]
    colbuffer[Fs, 1:npivots] .= 0.0
    put!(colchannel, colbuffer)

    return rows[1:npivots], cols[1:npivots]
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Int},
    s::Vector{Int},
    colchannel::Channel{K},
    rowbuffer::K;
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.iACA,RP}
    !isnothing(compressor.representor) && (s, Fs=compressor.representor(s, Fs))
    colbuffer = take!(colchannel)
    maxrank = min(maxrank, min(length(s), length(Fs)))
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    rowbuffer[1:maxrank, s] .= 0.0
    npivots, rows, cols = compressor.lrf(
        farmatrix,
        colbuffer,
        view(rowbuffer, 1:maxrank, s),
        maxrank;
        rows=rows,
        cols=cols,
        rowidcs=Fs,
        colidcs=s,
    )
    rowbuffer[1:npivots, s] = colbuffer[1:npivots, 1:npivots] * rowbuffer[1:npivots, s]
    colbuffer[1:npivots, 1:npivots] .= 0.0
    put!(colchannel, colbuffer)

    return Fs[rows], s[cols]
end
