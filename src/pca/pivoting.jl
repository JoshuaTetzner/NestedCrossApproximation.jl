struct PCAPivoting{F<:Real} <: FastBEAST.FD
    fct::Function
    weights::Vector{F}
    h::Vector{F}
    pos::Vector{SVector{3,F}}
end

function PCAPivoting(
    fct::Function, ref::SVector{3,F}, pos::Vector{SVector{3,F}};
) where {F<:Real}

    weights = ones(F, length(pos))
    for i in eachindex(pos)
        weights[i] = (norm(pos[i] - ref))
    end
    weights = fct.(weights)

    return PCAPivoting(fct, weights, zeros(F, length(pos)), pos)
end

#=
function PCAPivoting(ref::SVector{3,F}, pos::Vector{SVector{3,F}}) where {F<:Real}
    weights = ones(F, length(pos))
    for i in eachindex(pos)
        weights[i] = (norm(pos[i] - ref))
    end
    weights = 1 ./ (weights).^3 .+ 1 ./ (weights)
    return PCAPivoting(fct, weights, zeros(F, length(pos)), pos)
end=#

function PCAPivoting(fct::Function, pos::Vector{SVector{3,F}}) where {F<:Real}
    weights = ones(F, length(pos))

    return PCAPivoting(fct, weights, zeros(F, length(pos)), pos)
end

function weightedrandom(weights::Vector{F}) where F <: Real
    cs = cumsum(weights)
    rv = rand(minimum(weights)/2:(cs[end]/(10length(cs))):cs[end])
    for (ind, val) in enumerate(cs)
        if rv <= val
            return ind
            break
        end
    end
end

function FastBEAST.filldistance(
    fdmemory::PCAPivoting{F},
    usedidcs::Union{Vector{Bool},SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}},
) where {F<:Real}

    return argmax(fdmemory.h .* fdmemory.weights)
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
    firstidcs = argmax(pivstrat.weights)

    h = zeros(eltype(pivstrat.h), length(localpos))
    for i in eachindex(h)
        h[i] = norm(localpos[i] - localpos[firstidcs])
    end
    
    return PCAPivoting(pivstrat.fct, pivstrat.weights, h, localpos), firstidcs
end
