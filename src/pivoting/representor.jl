"""
    Representor

Abstract type for strategies that reduce the far field of a cluster to a few
representative points before the low-rank factorization.

A compressor with a representor factorizes the cluster against the representatives only,
which reduces the number of evaluated matrix entries at the cost of some accuracy.

# Interface

A representor is called as `representor(refidcs, idcs, npivots)` and returns at most
`npivots` of the candidates `idcs`; `refidcs` are the indices of the cluster.

# Concrete Types

  - [`RandomRep`](@ref): Random subset
  - [`ChebyshevRep`](@ref): Points close to a Chebyshev grid
  - [`MimicryRep`](@ref): Geometric selection by tree mimicry pivoting
"""
abstract type Representor end

"""
    RandomRep <: Representor

Representor that draws the representatives uniformly at random from the far field.
"""
struct RandomRep <: Representor end

function (::RandomRep)(::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int)
    npivots > 0 || throw(ArgumentError("npivots must be positive, got $npivots."))
    n = length(idcs)
    k = min(npivots, n)
    pool = collect(idcs)
    @inbounds for i in 1:k
        j = i + rand(0:(n - i))
        pool[i], pool[j] = pool[j], pool[i]
    end
    return pool[1:k]
end

"""
    ChebyshevRep{D,F} <: Representor

Representor that selects the points of the far field closest to a tensor Chebyshev grid.

The grid is aligned with the principal axes of the far field and scaled to its extent.

# Fields

  - `pos::Vector{SVector{D,F}}`: positions of the indices of the far field

# Notes

Only three-dimensional positions are supported.
"""
struct ChebyshevRep{D,F} <: Representor
    pos::Vector{SVector{D,F}}
end

#Covariance matrix (Bebendorf)
function covmat(M::AbstractVector{SVector{D,F}}) where {D,F}
    Cₜ = zeros(F, D, D)
    cₘ = sum(M) / length(M)

    for mᵢ in M
        Cₜ += (mᵢ - cₘ) * transpose((mᵢ - cₘ))
    end
    s, _, _ = svd(Cₜ)

    # Half extent along each principal axis, on either side of the centroid.
    axes = (SVector{D,F}(s[:, 1]), SVector{D,F}(s[:, 2]), SVector{D,F}(s[:, 3]))
    hs = zeros(F, 3)
    for mᵢ in M
        d = mᵢ - cₘ
        for a in 1:3
            h = abs(dot(d, axes[a]))
            hs[a] < h && (hs[a] = h)
        end
    end

    return s, hs, cₘ
end

function _chebyshevnodes1d(N::Int, ::Type{F}) where {F}
    return [cos(F((2i - 1) * π / (2N))) for i in 1:N]
end

# Smallest 3D tensor-product Chebyshev grid with at least `npivots` nodes.
function _chebyshevgrid3d(npivots::Int, ::Type{F}) where {F}
    N = max(1, ceil(Int, cbrt(npivots)))
    cn = _chebyshevnodes1d(N, F)
    return [SVector(cn[i], cn[j], cn[k]) for i in 1:N for j in 1:N for k in 1:N]
end

# The grid is fit to the candidates themselves, `refidcs` is not used.
function (cr::ChebyshevRep{D,F})(
    ::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int
) where {D,F}
    @assert D == 3 "ChebyshevRep currently only supports 3D point sets."
    npivots > 0 || throw(ArgumentError("npivots must be positive, got $npivots."))
    length(idcs) <= npivots && return collect(idcs)

    @views S, hs, cₘ = covmat(cr.pos[idcs])
    refnodes = _chebyshevgrid3d(npivots, F)
    @inbounds for n in eachindex(refnodes)
        refnodes[n] = S * (refnodes[n] .* hs) + cₘ
    end

    used = falses(length(idcs))
    chosen = Int[]
    sizehint!(chosen, npivots)
    @inbounds for refnode in refnodes
        length(chosen) == npivots && break
        best = 0
        bestdist = Inf
        for (nidx, node) in enumerate(idcs)
            used[nidx] && continue
            d = norm(cr.pos[node] - refnode)
            if d < bestdist
                bestdist = d
                best = nidx
            end
        end
        used[best] = true
        push!(chosen, idcs[best])
    end
    return chosen
end

"""
    MimicryRep{D,T,TreeType} <: Representor

Representor that selects the representatives with the tree mimicry pivoting, i.e. from the
geometry of the cluster and its far field.

# Fields

  - `pivoting::TreeMimicryPivoting{D,T,TreeType}`: pivoting strategy of
    AdaptiveCrossApproximation.jl, built from the positions of the cluster side, the
    positions of the far side, and the tree of the far side

# Notes

Requires a far field given as clusters and is therefore only supported with
[`BottomUp`](@ref).
"""
struct MimicryRep{D,T,TreeType} <: Representor
    pivoting::AdaptiveCrossApproximation.TreeMimicryPivoting{D,T,TreeType}
end

struct MimicryRepFunctor{D,T,TreeType} <: Representor
    pivoting::AdaptiveCrossApproximation.TreeMimicryPivotingFunctor{D,T,TreeType}
end

function (pr::MimicryRepFunctor)(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int
)
    npivots > 0 || throw(ArgumentError("npivots must be positive, got $npivots."))
    npivots = min(npivots, length(pr.pivoting.usedidcs))
    pivots = zeros(Int, npivots)
    reset!(pr.pivoting, refidcs, idcs)
    pivots[1] = pr.pivoting()
    for i in 2:npivots
        pivots[i] = pr.pivoting(i)
    end
    return pivots
end
