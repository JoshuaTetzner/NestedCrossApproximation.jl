struct RPivoting{I} <: FastBEAST.FD
    N::I
end

function RPivoting()
    return RPivoting(0)
end

function FastBEAST.firstpivot(pivstr::RPivoting{Int}, globalidcs::Vector{Int})
    return RPivoting(length(globalidcs)), rand(1:length(globalidcs))
end

function pivoting(
    pivstrat::NestedCrossApproximation.RPivoting{I},
    usedidcs::Union{SubArray{Bool, 1, Vector{Bool}, Tuple{UnitRange{Int64}}, true}, Vector{Bool}}
) where I
    nextidx = rand(1:pivstrat.N)
    if usedidcs[nextidx]
        return argmin(usedidcs)
    else
        return nextidx
    end
end

struct PCAPivoting2{F<:Real} <: FastBEAST.FD
    w::Vector{F}
    h::Vector{F}
    leja::Vector{F}
    pos::Vector{SVector{3,F}}
end

function PCAPivoting2(
    ref::SVector{3,F}, pos::Vector{SVector{3,F}};
) where {F<:Real}

    w = 1 ./ norm.(pos .- Scalar(ref))
    leja = ones(F, length(pos))
    h = zeros(F, length(pos))

    return PCAPivoting2(w, h, leja, pos)
end

function PCAPivoting2(
    pos::Vector{SVector{3,F}};
) where {F<:Real}

    w = ones(F, length(pos))
    leja = ones(F, length(pos))
    h = zeros(F, length(pos))

    return PCAPivoting2(w, h, leja, pos)
end

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
        weights[i] = fct(norm(pos[i] - ref))
    end

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
    fdmemory::PCAPivoting2{F},
    usedidcs::Union{Vector{Bool},SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}},
) where {F<:Real}
    nextidx = argmax(fdmemory.leja .^ (2/sum(usedidcs)) .* fdmemory.h .* fdmemory.w .^4)
    fdmemory.leja .*= norm.(fdmemory.pos .- Scalar(fdmemory.pos[nextidx]))
    return nextidx
end

function FastBEAST.filldistance(
    fdmemory::PCAPivoting{F},
    usedidcs::Union{Vector{Bool},SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}},
) where {F<:Real}

    return argmax(fdmemory.h .* fdmemory.weights)
end

function FastBEAST.filldistance(
    fdmemory::NestedCrossApproximation.PCAPivoting{F}
    #sedidcs::Union{Vector{Bool},SubArray{Bool,1,Vector{Bool},Tuple{UnitRange{Int}},true}},
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

function FastBEAST.firstpivot(pivstrat::PCAPivoting2{F}, globalidcs::Vector{Int}) where F

    firstidx = argmax(pivstrat.w)
    pivstrat.h .= norm.(pivstrat.pos .- Scalar(pivstrat.pos[firstidx]))
    pivstrat.leja .*= norm.(pivstrat.pos .- Scalar(pivstrat.pos[firstidx])) 

    return PCAPivoting2(pivstrat.w, pivstrat.h, pivstrat.leja, pivstrat.pos), firstidx
end
