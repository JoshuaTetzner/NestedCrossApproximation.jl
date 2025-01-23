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

function checktol(lm, aM, r, c)
    m = zeros(ComplexF64, size(lm, 1), size(lm, 2))
    @views lm.μ(m, lm.τ, lm.σ)
    println(length(r))
    return println(norm(m - aM) / norm(m))
end

function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)
    rbuffer .= 0
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    npivots = lrf(lm, rbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"
    checktol(
        lm,
        cbuffer[testidcs, 1:npivots] * rbuffer[1:npivots, 1:length(trialidcs)],
        rpivots,
        cpivots,
    )

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * rbuffer[1:npivots, cpivots]

    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:iACA}
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = init(compressor.lrf, lm)
    rbuffer .= 0
    cbuffer[testidcs, 1:maxrank] .= 0
    if maxrank > min(length(testidcs), length(trialidcs))
        maxrank = min(length(testidcs), length(trialidcs))
    end
    rpivots, cpivots = lrf(lm, rbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol)
    npivots = length(rpivots)
    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * rbuffer[1:npivots, 1:npivots]
    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::TopDownCompressor{CompressorType,RepresentorType})(
    cbuffer::Matrix{K},
    rbuffer::Matrix{K},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:LRF.ACA,RepresentorType<:Representor}
    trialidcs = compressor.representor(trialidcs)
    lm = FastBEAST.LRF.LazyMatrix(assembler, testidcs, trialidcs, K)
    lrf = LRF.init(compressor.lrf, lm)
    rbuffer .= 0
    cbuffer[testidcs, 1:maxrank] .= 0
    npivots = lrf(lm, rbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol)
    rpivots = LRF.rows(lrf)
    cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"
    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * rbuffer[1:npivots, cpivots[1:npivots]]
    return (testidcs[rpivots], trialidcs[cpivots])
end

#=
struct ACA{RepresentorType} <: TopDown
    representor::RepresentorType
end

function (::ACA{Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler,
    testidcs,
    trialidcs;
    tol=1e-4, maxrank=40
) where K
    lm = FastBEAST.LazyMatrix(assembler, testidcs, trialidcs, K)
    localrbuffer = take!(rbuffer)
    am = FastBEAST.ACAGlobalMemory(
        cbuffer[testidcs, 1:maxrank],
        localrbuffer,
        zeros(Bool, length(testidcs)),
        zeros(Bool, length(trialidcs)),
        0.0,
        0
    )

    U, V, rows, cols = aca(
        lm,
        am,
        tol = tol
    )

    @views cbuffer[testidcs, 1:length(cols)] =
        cbuffer[testidcs, 1:length(cols)] * localrbuffer[1:length(rows), cols]
    put!(rbuffer, localrbuffer)
    return testidcs[rows], trialidcs[cols]
end

function (aca::ACA{K})(x::Vector{Float64}) where K <: Representor
    return false
end
src/representor.jl

=#
