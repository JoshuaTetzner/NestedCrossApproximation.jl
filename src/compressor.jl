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

struct BottomUpCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function BottomUpCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function BottomUpCompressor(;
    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
)
    return BottomUpCompressor(factorization, representor)
end

function tolerance!(
    lrf::AdaptiveCrossApproximation.ACA{RP,CP,CC}, denominator::F
) where {RP,CP,CC<:FNormEstimator,F}
    return lrf.convergence.tol = lrf.convergence.tol / denominator
end

function tolerance!(
    lrf::AdaptiveCrossApproximation.iACA{RP,CP,CC}, denominator::F
) where {RP,CP,CC<:FNormExtrapolator,F}
    return lrf.convergence.estimator.tol = lrf.convergence.estimator.tol / denominator
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    t::Vector{Int},
    Ft::Vector{Int},
    colbuffer::K,
    rowchannel::Channel{K};
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.ACA,RP}
    !isnothing(compressor.representor) && (t, Ft=compressor.representor(t, Ft))
    rowbuffer = take!(rowchannel)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    colbuffer[t, 1:maxrank] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, t, 1:maxrank),
        view(rowbuffer, 1:maxrank, Ft),
        min(maxrank, min(length(t), length(Ft)));
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

    return rows, cols
end

function compress(
    compressor::TopDownCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Int},
    s::Vector{Int},
    colchannel::Channel{K},
    rowbuffer::K;
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.ACA,RP}
    !isnothing(compressor.representor) && (s, Fs=compressor.representor(s, Fs))
    colbuffer = take!(colchannel)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    rowbuffer[1:maxrank, s] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, Fs, 1:maxrank),
        view(rowbuffer, 1:maxrank, s),
        min(maxrank, min(length(s), length(Fs)));
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
    Fsvals::Vector{Int},
    svals::Vector{Int},
    colchannel::Channel{K},
    rowbuffer::K;
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:AdaptiveCrossApproximation.iACA,RP}
    !isnothing(compressor.representor) &&
        (svals, Fsvals=compressor.representor(svals, Fsvals))
    colbuffer = take!(colchannel)
    maxrank = min(maxrank, min(length(svals), length(Fsvals)))
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    rowbuffer[1:maxrank, svals] .= 0.0
    npivots, rows, cols = compressor.lrf(
        farmatrix,
        colbuffer,
        view(rowbuffer, 1:maxrank, svals),
        maxrank;
        rows=rows,
        cols=cols,
        rowidcs=Fsvals,
        colidcs=svals,
    )
    rowbuffer[1:npivots, svals] =
        colbuffer[1:npivots, 1:npivots] * rowbuffer[1:npivots, svals]
    colbuffer[1:npivots, 1:npivots] .= 0.0
    put!(colchannel, colbuffer)

    return rows, cols
end
