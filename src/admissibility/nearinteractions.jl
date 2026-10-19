import H2Trees: halfsize, radius, center, isleaf
import AdaptiveCrossApproximation:
    IsNearFunctor, isnear, nearinteractions, assemblenears, PreserveSpaceOrder

"""
    IsNearWidebandFunctor{F,LF}

Admissibility predicate for wideband (mixed low-/high-frequency) hierarchical
matrices.

Classifies a pair of tree nodes as near (inadmissible for compression, must be
assembled directly) or far (admissible). The criterion switches per pair between
a geometric one in the low-frequency regime and a wave-based one in the
high-frequency regime, as decided by the `islf` field.

Construct with [`isnearwideband`](@ref) rather than calling the constructor
directly.

# Fields

  - `k::F`: wavenumber.
  - `ηlf::F`: admissibility parameter used in the low-frequency branch.
  - `ηhf::F`: admissibility parameter used in the high-frequency branch.
  - `islf::LF`: low-frequency classifier, e.g. an [`IsLowFrequencyFunctor`](@ref).
"""
struct IsNearWidebandFunctor{F,LF}
    k::F
    ηlf::F
    ηhf::F
    islf::LF
end

"""
    isnearwideband(k::Real; ηlf=1.0, ηhf=5.0, γ=1.0, islf=islf(γ, k))

Create a wideband admissibility predicate for hierarchical matrix construction.

Two nodes are classified as near when, depending on whether their size places
them in the low- or high-frequency regime (as decided by `islf`), they fail the
corresponding geometric (`ηlf`) or wave-based (`ηhf`) separation criterion for
wavenumber `k`.

All parameters are stored as `float(typeof(k))`, so the wavenumber alone fixes
the predicate's element type.

# Arguments

  - `k::Real`: wavenumber.
  - `ηlf::Real = 1.0`: admissibility parameter for the low-frequency branch.
  - `ηhf::Real = 5.0`: admissibility parameter for the high-frequency branch.
  - `γ::Real = 1.0`: threshold used by the default low-frequency classifier
    (see [`islf`](@ref)); ignored if `islf` is passed explicitly.
  - `islf = islf(γ, k)`: low-frequency classifier deciding, per node, which
    branch applies.

# Returns

  - [`IsNearWidebandFunctor`](@ref).
"""
function isnearwideband(
    k::F; ηlf::Real=F(1.0), ηhf::Real=F(5.0), γ::Real=F(1.0), islf::Any=islf(γ, k)
) where {F<:Real}
    FT = float(F)
    return IsNearWidebandFunctor{FT,typeof(islf)}(FT(k), FT(ηlf), FT(ηhf), islf)
end

function (isnear::IsNearWidebandFunctor{F})(
    treea::H2Trees.TwoNTree, treeb::H2Trees.TwoNTree, nodea::Int, nodeb::Int
) where {F}
    ths = halfsize(treea, nodea) * sqrt(F(3))
    shs = halfsize(treeb, nodeb) * sqrt(F(3))
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

"""
    IsLowFrequencyFunctor{F}

Classifier deciding whether a tree node is in the low-frequency regime.

A node is low-frequency when `k * d <= γ`, where `d` is a characteristic
diameter: a plain value when called as `f(d)`, or one derived from a tree node
when called as `f(tree, node)`.

Construct with [`islf`](@ref) rather than calling the constructor directly.

# Fields

  - `γ::F`: admissibility threshold.
  - `k::F`: wavenumber.
"""
struct IsLowFrequencyFunctor{F}
    γ::F
    k::F
end

wavenumber(f::IsLowFrequencyFunctor) = f.k

"""
    islf(γ::Real, k::Real)

Create a low-frequency classifier for wavenumber `k` and threshold `γ`.

# Returns

  - [`IsLowFrequencyFunctor`](@ref).
"""
function islf(γ::Real, k::Real)
    F = float(typeof(k))
    return IsLowFrequencyFunctor{F}(F(γ), F(k))
end

@inline function (f::IsLowFrequencyFunctor)(diam::Real)
    return f.k * diam <= f.γ * one(promote_type(typeof(f.k), typeof(diam)))
end

@inline function (f::IsLowFrequencyFunctor{F})(tree::H2Trees.TwoNTree, level::Int) where {F}
    # Characteristic box diameter at this level in TwoN trees.
    rootlevel = H2Trees.minimumlevel(tree)
    N = size(eltype(tree), 1)
    diam = 2 * sqrt(F(N)) * H2Trees.halfsize(tree) / F(2)^(level - rootlevel)
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
