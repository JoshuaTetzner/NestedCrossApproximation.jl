struct TopDownCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function TopDownCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function TopDownCompressor(; factorization=LRF.ACA(), representor=nothing)
    return TopDownCompressor(factorization, representor)
end

#compression separate for test and trial tree...
#Standard NCA
function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)
    localrbuffer = take!(rbuffer)
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    npivots = lrf(lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, cpivots]

    localrbuffer[1:npivots, 1:length(trialidcs)] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)

    localcbuffer = take!(cbuffer)
    rbuffer[1:maxrank, trialidcs] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    npivots = lrf(lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[rpivots, 1:npivots] * rbuffer[1:npivots, trialidcs]

    localcbuffer[1:length(testidcs), 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end

#iACA
function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:iACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = init(compressor.lrf, lm)
    localrbuffer = take!(rbuffer)
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = lrf(
        lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol
    )
    npivots = length(rpivots)

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, 1:npivots]
    localrbuffer[1:npivots, 1:npivots] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:iACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = init(compressor.lrf, lm)
    rbuffer[1:maxrank, trialidcs] .= 0
    localcbuffer = take!(cbuffer)
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = lrf(
        lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol
    )
    npivots = length(rpivots)

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[1:npivots, 1:npivots] * rbuffer[1:npivots, trialidcs]

    localcbuffer[1:npivots, 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,RepresentorType})(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA,RepresentorType<:Representor}
    trialidcs = compressor.representor(trialidcs)

    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)

    localrbuffer = take!(rbuffer)
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    cbuffer[testidcs, 1:maxrank] .= 0

    npivots = lrf(lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, cpivots[1:npivots]]

    localrbuffer[1:npivots, 1:length(trialidcs)] .= 0
    put!(rbuffer, localrbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,RepresentorType})(
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA,RepresentorType<:Representor}
    testidcs = compressor.representor(testidcs)

    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)

    localcbuffer = take!(cbuffer)

    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rbuffer[1:maxrank, trialidcs] .= 0

    npivots = lrf(lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[rpivots, 1:npivots] * rbuffer[1:npivots, trialidcs]

    localcbuffer[1:length(testidcs), 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end
