abstract type Representor end

# All representors are called as `representor(refidcs, idcs, npivots) -> Vector{Int}`:
# pick at most `npivots` representative point indices from the candidate set `idcs`,
# optionally guided by the reference point set `refidcs` (the cluster the result will
# stand in for). They reduce a far-field index set before it's handed to the low-rank
# factorization, trading some accuracy for a smaller, cheaper pivot search.

struct RandomRep <: Representor end

function (::RandomRep)(::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int)
    n = length(idcs)
    k = min(npivots, n)
    pool = collect(idcs)
    @inbounds for i in 1:k
        j = i + rand(0:(n - i))
        pool[i], pool[j] = pool[j], pool[i]
    end
    return pool[1:k]
end

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

    hs = zeros(F, 3)
    for mᵢ in M
        h1 = dot((mᵢ - cₘ), s[1:3, 1])
        h2 = dot((mᵢ - cₘ), s[1:3, 2])
        h3 = dot((mᵢ - cₘ), s[1:3, 3])
        if (hs[1] < h1)
            hs[1] = h1
        end
        if (hs[2] < h2)
            hs[2] = h2
        end
        if (hs[3] < h3)
            hs[3] = h3
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

# refidcs is unused: unlike MimicryRep, the Chebyshev grid is fit to the candidate set's
# own spatial distribution (via covmat), not mimicking a separate reference distribution.
function (cr::ChebyshevRep{D,F})(
    ::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int
) where {D,F}
    @assert D == 3 "ChebyshevRep currently only supports 3D point sets."
    npivots <= 0 && return Int[]
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

struct MimicryRep{D,T,TreeType} <: Representor
    pivoting::AdaptiveCrossApproximation.TreeMimicryPivoting{D,T,TreeType}
end

struct MimicryRepFunctor{D,T,TreeType} <: Representor
    pivoting::AdaptiveCrossApproximation.TreeMimicryPivotingFunctor{D,T,TreeType}
end

function (pr::MimicryRepFunctor)(
    refidcs::AbstractVector{Int}, idcs::AbstractVector{Int}, npivots::Int
)
    npivots = min(npivots, length(pr.pivoting.usedidcs))
    pivots = zeros(Int, npivots)
    reset!(pr.pivoting, refidcs, idcs)
    pivots[1] = pr.pivoting()
    for i in 2:npivots
        pivots[i] = pr.pivoting(i)
    end
    return pivots
end
