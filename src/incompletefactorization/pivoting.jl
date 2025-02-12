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
