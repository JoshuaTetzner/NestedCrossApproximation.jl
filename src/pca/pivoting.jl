struct PCAPivoting{F<:Real} <: FastBEAST.FD
    weights::Vector{F}
    h::Vector{F}
    pos::Vector{SVector{3,F}}
end

function PCAPivoting(ref::SVector{3,F}, pos::Vector{SVector{3,F}}) where {F<:Real}
    weights = zeros(F, length(pos))
    for i in eachindex(pos)
        weights[i] = (norm(pos[i] - ref))
    end
    weights = weights .^ -2 * maximum(weights)
    return PCAPivoting(weights, zeros(F, length(pos)), pos)
end

function PCAPivoting(pos::Vector{SVector{3,F}}) where {F<:Real}
    weights = zeros(F, length(pos))

    return PCAPivoting(weights, zeros(F, length(pos)), pos)
end

function FastBEAST.filldistance(
    fdmemory::PCAPivoting{F},
    usedidcs::Union{Vector{Bool},SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}},
) where {F<:Real}
    h = fdmemory.h .* fdmemory.weights
    return [argmax(h)]
end

"""
    function firstpivot(pivstrat::FD, globalidcs::Vector{Int})

Returns first index of the pivoting strategy. For `FillDistance` this will be the
basis function closest to the center of the distribution.

# Arguments

  - `pivstrat::FD`: Pivoting strategy.
  - `globalidcs::Vector{Int}`: Indices corresponding to the matrix block, used to determine the
    basis functions/positions used for pivoting.
"""
function FastBEAST.firstpivot(pivstrat::PCAPivoting{F}, globalidcs::Vector{Int}) where F

    localpos = pivstrat.pos
    center = sum(localpos) / length(localpos)

    firstidcs = 0
    minimum = 0
    for (ind, pos) in enumerate(localpos)
        if ind == 1 || minimum > norm(pos - center)
            firstidcs = ind
            minimum = norm(pos-center)
        end
    end

    h = zeros(eltype(pivstrat.h), length(localpos))
    for i in eachindex(h)
        h[i] = norm(localpos[i] - localpos[firstidcs])
    end
    
     return PCAPivoting(pivstrat.weights, h, localpos), firstidcs
end
