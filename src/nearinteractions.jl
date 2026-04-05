import H2Trees: isleaf, testtree, trialtree, root, children, halfsize, radius, center

struct IsNearFunctor{F}
    η::F
end

"""
Low-frequency classifier based on diameter and wavenumber.

`islf(d)` returns true when `k * d <= 1`.
"""
struct IsLowFrequencyFunctor{F}
    γ::F
    k::F
end

wavenumber(f::IsLowFrequencyFunctor) = f.k

function islf(γ::Real, k::Real)
    F = float(typeof(k))
    return IsLowFrequencyFunctor{F}(F(γ), F(k))
end

@inline function (f::IsLowFrequencyFunctor)(diam::Real)
    return f.k * diam <= f.γ * one(promote_type(typeof(f.k), typeof(diam)))
end

@inline function (f::IsLowFrequencyFunctor)(tree::H2Trees.TwoNTree, level::Int)
    # Characteristic box diameter at this level in TwoN trees.
    diam = 2 * sqrt(3) * H2Trees.halfsize(tree) / (2.0^(level - 1))
    return f.k * diam <= f.γ * one(promote_type(typeof(f.k), typeof(diam)))
end

# Generalized node radius for BoundingBallTree.
# This enforces identical LF/HF classification for all siblings.
function _bbgenradius(tree::H2Trees.BoundingBallTree, node::Int)
    pnode = H2Trees.parent(tree, node)
    pnode == 0 && return H2Trees.radius(tree, node)
    sibs = collect(H2Trees.children(tree, pnode))
    avg = mapreduce(c -> H2Trees.radius(tree, c), +, sibs) / length(sibs)
    return min(_bbgenradius(tree, pnode), avg)
end

@inline function (f::IsLowFrequencyFunctor)(tree::H2Trees.BoundingBallTree, node::Int)
    return f(2 * _bbgenradius(tree, node))
end

"""
HF/LF-aware near criterion for wideband settings.

  - LF branch: geometric admissibility with `ηlf`.
  - HF branch: wave admissibility with `ηhf` and `k`.
"""
struct IsNearWidebandFunctor{F,LF}
    k::F
    ηlf::F
    ηhf::F
    islf::LF
end

function isnearwideband(
    k::Real; ηlf::Real=1.0, ηhf::Real=5.0, γ::Real=1.0, islf::Any=islf(γ, k)
)
    F = promote_type(float(typeof(k)), float(typeof(ηlf)), float(typeof(ηhf)))
    return IsNearWidebandFunctor{F,typeof(islf)}(F(k), F(ηlf), F(ηhf), islf)
end

function isnear(η::Real=Float64(1.0))
    return IsNearFunctor{typeof(η)}(η)
end

function (isnear::IsNearFunctor{F})(
    treea::TwoNTree, treeb::TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = halfsize(treea, nodea) * sqrt(3)
    shs = halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::IsNearFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = radius(treea, nodea)
    shs = radius(treeb, nodeb)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)

    (2 * max(ths, shs) <= isnear.η * max(dist, 0.0)) ? (return false) : (return true)
end

function (isnear::IsNearWidebandFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = halfsize(treea, nodea) * sqrt(3)
    shs = halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)
    sep = max(dist, zero(dist))

    if isnear.islf(min(2 * ths, 2 * shs))
        return 2 * max(ths, shs) > isnear.ηlf * sep
    end
    return 4 * isnear.k * max(ths^2, shs^2) > isnear.ηhf * sep
end

function (isnear::IsNearWidebandFunctor{F})(
    treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int
) where {F}
    ths = radius(treea, nodea)
    shs = radius(treeb, nodeb)
    dist = norm(center(treea, nodea) - center(treeb, nodeb)) - (ths + shs)
    sep = max(dist, zero(dist))
    if isnear.islf(min(2 * ths, 2 * shs))
        return 2 * max(ths, shs) > isnear.ηlf * sep
    end
    return 4 * isnear.k * max(ths^2, shs^2) > isnear.ηhf * sep
end

function nears!(
    tree,
    values::Vector{V},
    nearvalues::Vector{V},
    tnode::Int,
    snodes::V;
    isnear=H2Trees.isnear,
) where {V<:Vector{Int}}
    nearnodes = Int[]
    childnearnodes = Int[]
    for snode in snodes
        if isnear(testtree(tree), trialtree(tree), tnode, snode)
            if isleaf(testtree(tree), tnode) || isleaf(trialtree(tree), snode)
                push!(nearnodes, snode)
            else
                append!(childnearnodes, collect(children(trialtree(tree), snode)))
            end
        end
    end
    if nearnodes != []
        push!(nearvalues, H2Trees.values(trialtree(tree), nearnodes))
        push!(values, H2Trees.values(testtree(tree), tnode))
    end
    if childnearnodes != []
        for child in children(testtree(tree), tnode)
            nears!(tree, values, nearvalues, child, childnearnodes; isnear=isnear)
        end
    end
end

function nearinteractions(tree::BlockTree; isnear=H2Trees.isnear)
    !isnear(testtree(tree), trialtree(tree), root(testtree(tree)), root(trialtree(tree))) &&
        return Vector{Int}(), Vector{Int}[]
    values = Vector{Int}[]
    nearvalues = Vector{Int}[]
    nears!(
        tree,
        values,
        nearvalues,
        root(testtree(tree)),
        [root(trialtree(tree))];
        isnear=isnear,
    )
    return values, nearvalues
end

function assemblenears(
    operator,
    testspace,
    trialspace,
    tree;
    isnear=isnear(),
    scheduler=SerialScheduler(),
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
)
    nearmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=matrixdata
    )
    values, nearvalues = nearinteractions(tree; isnear=isnear)
    if isempty(values)
        return BlockSparseMatrix(
            Matrix{eltype(nearmatrix)}[], Vector{Int}[], Vector{Int}[], size(nearmatrix)
        )
    end
    blocks = zeros.(eltype(nearmatrix), length.(values), length.(nearvalues))
    @tasks for i in eachindex(blocks)
        @set scheduler = scheduler
        nearmatrix(blocks[i], values[i], nearvalues[i])
    end
    nears = BlockSparseMatrix(
        blocks, values, nearvalues, size(nearmatrix); scheduler=scheduler
    )
    return nears
end
