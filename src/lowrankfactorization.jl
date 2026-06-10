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

struct BottomUpCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function BottomUpCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function BottomUpCompressor(; factorization=LRF.ACA(), representor=nothing)
    return BottomUpCompressor(factorization, representor)
end

#Standard NCA
function (compressor::Union{TopDownCompressor{CompressorType,Nothing}})(
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

function (compressor::Union{TopDownCompressor{CompressorType,Nothing}})(
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

function (compressor::BottomUpCompressor{CompressorType,TreeMimicryRepresentor})(
    tree::NminTree{D},
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    node::Int,
    fars::Vector{Int},
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:LRF.ACA}
    localrbuffer = take!(rbuffer)
    if ClusterTrees.haschildren(tree, node)
        rowidcs = Int[]
        for child in ClusterTrees.children(tree, node)
            append!(rowidcs, pivots[child][1])
        end
    else
        rowidcs = value(tree, node)
    end
    cbuffer[rowidcs, 1:maxrank] .= 0.0

    colidcs = compressor.representor(node, fars, length(rowidcs))

    lm = FastBEAST.LRF.LazyMatrix(assembler, rowidcs, colidcs, K)
    lrf = LRF.init(compressor.lrf, lm)

    if maxrank > min(length(rowidcs), length(colidcs))
        maxrank = min(length(rowidcs), length(colidcs))
    end
    rpivots, cpivots, npivots = lrf(
        lm, localrbuffer, view(cbuffer, rowidcs, 1:maxrank), maxrank, tol
    )

    rpivots = rpivots[1:npivots]
    cpivots = cpivots[1:npivots]

    cbuffer[rowidcs, 1:npivots] =
        cbuffer[rowidcs, 1:npivots] * localrbuffer[1:npivots, cpivots]

    localrbuffer[1:npivots, 1:length(colidcs)] .= 0
    put!(rbuffer, localrbuffer)
    return (rowidcs[rpivots], colidcs[cpivots])
end

function (compressor::Union{BottomUpCompressor{CompressorType,TreeMimicryRepresentor}})(
    tree::NminTree{D},
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    fars::Vector{Int},
    node::Int,
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:LRF.ACA}
    localcbuffer = take!(cbuffer)
    if ClusterTrees.haschildren(tree, node)
        colidcs = Int[]
        for child in ClusterTrees.children(tree, node)
            append!(colidcs, pivots[child][2])
        end
    else
        colidcs = value(tree, node)
    end
    rbuffer[1:maxrank, colidcs] .= 0.0

    rowidcs = compressor.representor(node, fars, length(colidcs))

    lm = FastBEAST.LRF.LazyMatrix(assembler, rowidcs, colidcs, K)
    lrf = LRF.init(compressor.lrf, lm)

    if maxrank > min(length(rowidcs), length(colidcs))
        maxrank = min(length(rowidcs), length(colidcs))
    end
    rpivots, cpivots, npivots = lrf(
        lm, view(rbuffer, 1:maxrank, colidcs), localcbuffer, maxrank, tol
    )

    rpivots = rpivots[1:npivots]
    cpivots = cpivots[1:npivots]

    rbuffer[1:npivots, colidcs] =
        localcbuffer[rpivots, 1:npivots] * rbuffer[1:npivots, colidcs]

    localcbuffer[1:length(rowidcs), 1:npivots] .= 0
    put!(cbuffer, localcbuffer)
    return (rowidcs[rpivots], colidcs[cpivots])
end

#IACA
function (compressor::TopDownCompressor{CompressorType,Nothing})(
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    testidcs::Vector{Int},
    trialidcs::Vector{Int};
    tol=1e-4,
    maxrank=40,
) where {K,CompressorType<:IACA}
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

function (compressor::BottomUpCompressor{CompressorType,Nothing})(
    tree::NminTree{D},
    cbuffer::Matrix{K},
    rbuffer::Channel{Matrix{K}},
    assembler::Function,
    node::Int,
    fars::Vector{Int},
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:IACA}
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
) where {K,CompressorType<:IACA}
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

function (compressor::BottomUpCompressor{CompressorType,Nothing})(
    tree::NminTree{D},
    cbuffer::Channel{Matrix{K}},
    rbuffer::Matrix{K},
    assembler::Function,
    fars::Vector{Int},
    node::Int,
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    tol=1e-4,
    maxrank=40,
) where {D,K,CompressorType<:IACA}
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
