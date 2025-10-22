abstract type GeoPivStrat <: LRF.AbstractFillDistance end

mutable struct IACAPivoting{D,F} <: GeoPivStrat
    pos::Vector{SVector{D,F}}
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

function IACAPivoting(pos::Vector{SVector{D,F}}) where {D,F<:Real}
    return IACAPivoting(pos, F[], F[], F[])
end

function (pivstrat::IACAPivoting{D,F})(
    rcp::Vector{Int}; ref=SVector(F(0.0), F(0.0), F(0.0))
) where {D,F}
    h = zeros(F, length(rcp))
    w = 1 ./ norm.(pivstrat.pos[rcp] .- Scalar(ref))
    leja = ones(F, length(rcp))

    return IACAPivoting(pivstrat.pos[rcp], h, leja, w)
end

function (pivstrat::IACAPivoting{D,F})() where {D,F<:Real}
    nextidx = argmax(pivstrat.w)
    @views pivstrat.h .= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    pivstrat.leja .*= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    return nextidx
end

function (pivstrat::IACAPivoting{D,F})(npivot::Int) where {D,F<:Real}
    nextidx = argmax(pivstrat.leja .^ (2 / (npivot - 1)) .* pivstrat.h .* pivstrat.w .^ 4)
    LRF.filldistance!(pivstrat, nextidx)
    pivstrat.leja .*= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    return nextidx
end

mutable struct IACAPivotingMFIE{D,F} <: GeoPivStrat
    refpos::Vector{SVector{D,F}}
    pos::Vector{SVector{D,F}}
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

function IACAPivotingMFIE(refpos::Vector{SVector{D,F}}, pos) where {D,F<:Real}
    return IACAPivotingMFIE(refpos, pos, F[], F[], F[])
end

function (pivstrat::IACAPivotingMFIE{D,F})(
    rcp::Vector{Int}, refidcs::Vector{Int}; ref=SVector(F(0.0), F(0.0), F(0.0))
) where {D,F}
    h = zeros(F, length(rcp))

    leja = ones(F, length(rcp))

    w = zeros(F, length(rcp))
    pos = zeros(F, 3, length(refidcs))
    mp = mean(pivstrat.refpos[refidcs])
    for idx in eachindex(refidcs)
        pos[:, idx] = pivstrat.refpos[refidcs][idx] - mp
    end
    u, _, _ = svd(pos)

    for (idx, rc) in enumerate(rcp)
        d = abs(dot((pivstrat.pos[rc] - mp), u[:, 3]))
        if isapprox(d, 0.0; atol=1e-10)
            w[idx] = 0.0
        else
            w[idx] = 1
        end
    end
    w = w .* 1 ./ norm.(pivstrat.pos[rcp] .- Scalar(ref))
    return IACAPivotingMFIE(pivstrat.refpos[refidcs], pivstrat.pos[rcp], h, leja, w)
end

function (pivstrat::IACAPivotingMFIE{D,F})() where {D,F<:Real}
    nextidx = argmax(pivstrat.w)
    @views pivstrat.h .= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    pivstrat.leja .*= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    return nextidx
end

function (pivstrat::IACAPivotingMFIE{D,F})(npivot::Int) where {D,F<:Real}
    nextidx = argmax(pivstrat.leja .^ (2 / (npivot - 1)) .* pivstrat.h .* pivstrat.w .^ 4)
    LRF.filldistance!(pivstrat, nextidx)
    pivstrat.leja .*= norm.(pivstrat.pos .- Scalar(pivstrat.pos[nextidx]))
    return nextidx
end

struct IACAPivoting2{D,T,F} <: GeoPivStrat
    tree::NminTree{T}
    pos::Vector{SVector{D,F}}
end

struct HoldMyData{F}
    h::Vector{F}
    leja::Vector{F}
    w::Vector{F}
end

function findtarget(tree::NminTree{T}, node::Int, ref::SVector{3,F}) where {F,T}
    idx = tree.nodes[node].node.first_child
    d = norm(ClusterTrees.data(tree, idx).ct - ref)
    nextsibling = tree.nodes[node].node.next_sibling
    while nextsibling != 0
        if d > norm(ClusterTrees.data(tree, nextsibling).ct - ref)
            idx = nextsibling
            d = norm(ClusterTrees.data(tree, nextsibling).ct - ref)
        end
        nextsibling = tree.nodes[nextsibling].node.next_sibling
    end

    if ClusterTrees.haschildren(tree, idx)
        return findtarget(tree, idx, ref)
    else
        return idx
    end
end

function findtarget(
    pivstrat::IACAPivoting2{D,T,F}, node::Int, ref::SVector{D,F}, usedidcs::Vector{Int}
) where {D,T,F}
    childnodes = zeros(Int, pivstrat.tree.nodes[node].node.num_children)
    w = zeros(Float64, pivstrat.tree.nodes[node].node.num_children)
    h = zeros(Float64, pivstrat.tree.nodes[node].node.num_children)
    leja = ones(Float64, pivstrat.tree.nodes[node].node.num_children)
    idx = 1
    for child in ClusterTrees.children(pivstrat.tree, node)
        childnodes[idx] = child
        w[idx] = 1 / norm(ClusterTrees.data(pivstrat.tree, child).ct - ref)
        for uidx in usedidcs
            h[idx] += norm(ClusterTrees.data(pivstrat.tree, child).ct - pivstrat.pos[uidx])
            leja[idx] *= norm(
                ClusterTrees.data(pivstrat.tree, child).ct - pivstrat.pos[uidx]
            )
        end
        idx += 1
    end

    ch = childnodes[argmax(leja .^ (2 / length(usedidcs)) .* h .* w .^ 4)]

    if ClusterTrees.haschildren(pivstrat.tree, ch)
        return findtarget(pivstrat, ch, ref, usedidcs)
    else
        return ch
    end
end

function geopivoting(
    ref::SVector{3,F}, usedidcs::Vector{Int}, idcs::Vector{Int}, pos::Vector{SVector{3,F}}
) where {F}
    h = zeros(Float64, length(idcs))
    leja = ones(Float64, length(idcs))
    w = 1 ./ norm.(pos[idcs] .- Scalar(ref))
    for uidx in usedidcs
        leja .*= norm.(pos[idcs] .- Scalar(pos[uidx]))
    end
    h .= norm.(pos[idcs] .- Scalar(pos[usedidcs[1]]))
    for uidx in usedidcs[2:end]
        for i in eachindex(h)
            if h[i] > norm(pos[idcs[i]] - pos[uidx])
                h[i] = norm(pos[idcs[i]] - pos[uidx])
            end
        end
    end

    return idcs[argmax(leja .^ (2 / length(usedidcs)) .* h .* w .^ 4)]
end

function (pivstrat::IACAPivoting2{D,T,F})(
    ref::SVector{3,F}, fars::Vector{Int}, cts::Vector{SVector{3,F}}
) where {D,T,F}
    w = 1 ./ norm.(cts .- Scalar(ref))
    target = fars[argmax(w)]
    if ClusterTrees.haschildren(pivstrat.tree, target)
        target = findtarget(pivstrat.tree, target, ref)
    end

    distance = 0.0
    firstidx = 0
    for idx in value(pivstrat.tree, target)
        if norm(pivstrat.pos[idx] - ref) < distance || distance == 0.0
            distance = norm(pivstrat.pos[idx] - ref)
            firstidx = idx
        end
    end

    h = norm.(cts .- Scalar(pivstrat.pos[firstidx]))
    leja = copy(h)

    return firstidx, HoldMyData(h, leja, w)
end

function (pivstrat::IACAPivoting2{D,T,F})(
    ref::SVector{3,F},
    fars::Vector{Int},
    cts::Vector{SVector{3,F}},
    usedidcs::Vector{Int},
    data::HoldMyData{F},
) where {D,T,F}
    target = fars[argmax(data.leja .^ (2 / length(usedidcs)) .* data.h .* data.w .^ 4)]
    if ClusterTrees.haschildren(pivstrat.tree, target)
        target = findtarget(pivstrat, target, ref, usedidcs)
    end

    nextidx = geopivoting(ref, usedidcs, value(pivstrat.tree, target), pivstrat.pos)

    data.leja .*= norm.(cts .- Scalar(pivstrat.pos[nextidx]))
    for (i, ct) in enumerate(cts)
        if data.h[i] > norm(ct - pivstrat.pos[nextidx])
            data.h[i] = norm(ct - pivstrat.pos[nextidx])
        end
    end

    return nextidx, data
end
