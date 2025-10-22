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

struct TopDownCompressor2{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function TopDownCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

#function TopDownCompressor(;
#    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
#)
#    return TopDownCompressor(factorization, representor)
#end

struct ButtomUpCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function ButtomUpCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function ButtomUpCompressor(; factorization=LRF.ACA(), representor=nothing)
    return ButtomUpCompressor(factorization, representor)
end

struct ZhaoCompressor{LowRankFactorizationType}
    lrf::LowRankFactorizationType
end

function ZhaoCompressor(; factorization=LRF.ACA())
    return ZhaoCompressor(factorization)
end

#compression separate for test and trial tree...
#Standard NCA
function (
    compressor::Union{
        TopDownCompressor{CompressorType,Nothing},ZhaoCompressor{CompressorType}
    }
)(
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
    rpivots, cpivots, npivots = lrf(
        lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol
    )

    rpivots = rpivots[1:npivots]
    cpivots = cpivots[1:npivots]

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, cpivots]

    localrbuffer[1:npivots, 1:length(trialidcs)] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], trialidcs[cpivots])
end

function (
    compressor::Union{
        TopDownCompressor{CompressorType,Nothing},ZhaoCompressor{CompressorType}
    }
)(
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
    rpivots, cpivots, npivots = lrf(
        lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol
    )
    if rpivots[npivots] == rpivots[npivots - 1] || cpivots[npivots] == cpivots[npivots - 1]
        npivots -= 1
    end
    rpivots = rpivots[1:npivots]
    cpivots = cpivots[1:npivots]
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

    if rpivots[npivots] == rpivots[npivots - 1] || cpivots[npivots] == cpivots[npivots - 1]
        npivots -= 1
    end

    cbuffer[testidcs, 1:npivots] =
        cbuffer[testidcs, 1:npivots] * localrbuffer[1:npivots, 1:npivots]
    localrbuffer[1:npivots, 1:npivots] .= 0

    put!(rbuffer, localrbuffer)
    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::ButtomUpCompressor{CompressorType,Nothing})(
    tree::NminTree{D},
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    node::Int,
    fars::Vector{Int},
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:iACA}
    localrbuffer = take!(rbuffer)
    if ClusterTrees.haschildren(tree, node)
        rowidcs = Int[]
        for child in ClusterTrees.children(tree, node)
            append!(rowidcs, pivots[child][1])
        end
    else
        rowidcs = value(tree, node)
    end
    cbuffer[rowidcs, 1:maxrank] .= 0
    rpivots, cpivots = compressor.lrf(
        assembler,
        localrbuffer,
        view(cbuffer, rowidcs, 1:maxrank),
        rowidcs,
        maxrank,
        tol,
        tree.nodes[node].node.data.ct,
        fars,
    )
    npivots = length(rpivots)
    cbuffer[rowidcs, 1:npivots] =
        cbuffer[rowidcs, 1:npivots] * localrbuffer[1:npivots, 1:npivots]
    localrbuffer[1:npivots, 1:npivots] .= 0

    put!(rbuffer, localrbuffer)
    return (rpivots, cpivots)
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

    if rpivots[npivots] == rpivots[npivots - 1] || cpivots[npivots] == cpivots[npivots - 1]
        #println("fail")
        #println(compressor.lrf.rowpivoting.usedidcs)
        #error()
        npivots -= 1
    end

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[1:npivots, 1:npivots] * rbuffer[1:npivots, trialidcs]

    localcbuffer[1:npivots, 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end

function (compressor::ButtomUpCompressor{CompressorType,Nothing})(
    tree::NminTree{D},
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    fars::Vector{Int},
    node::Int,
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:iACA}
    localcbuffer = take!(cbuffer)
    if ClusterTrees.haschildren(tree, node)
        colidcs = Int[]
        for child in ClusterTrees.children(tree, node)
            append!(colidcs, pivots[child][2])
        end
    else
        colidcs = value(tree, node)
    end
    rbuffer[1:maxrank, colidcs] .= 0

    rpivots, cpivots = compressor.lrf(
        assembler,
        view(rbuffer, 1:maxrank, colidcs),
        localcbuffer,
        colidcs,
        maxrank,
        tol,
        tree.nodes[node].node.data.ct,
        fars,
    )
    npivots = length(rpivots)

    rbuffer[1:npivots, colidcs] =
        localcbuffer[1:npivots, 1:npivots] * rbuffer[1:npivots, colidcs]

    localcbuffer[1:npivots, 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (rpivots, cpivots)
end

#

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

    rpivots, cpivots, npivots = lrf(
        lm, localrbuffer, view(cbuffer, testidcs, 1:maxrank), maxrank, tol
    )
    #rpivots = LRF.rows(lrf)
    #cpivots = LRF.cols(lrf)
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

    rpivots, cpivots, npivots = lrf(
        lm, view(rbuffer, 1:maxrank, trialidcs), localcbuffer, maxrank, tol
    )
    #rpivots = LRF.rows(lrf)
    #cpivots = LRF.cols(lrf)
    npivots != length(rpivots) && @warn "ACA compression found zero rows or columns!"

    rbuffer[1:npivots, trialidcs] =
        localcbuffer[rpivots, 1:npivots] * rbuffer[1:npivots, trialidcs]

    localcbuffer[1:length(testidcs), 1:npivots] .= 0
    put!(cbuffer, localcbuffer)

    return (testidcs[rpivots], trialidcs[cpivots])
end
