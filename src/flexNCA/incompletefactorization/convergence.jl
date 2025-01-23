mutable struct IncompleteNormEstimator{F} <: LRF.ConvCrit
    lastnorms::Vector{F}
    normUV::F
end

(::IncompleteNormEstimator{F})(kwargs...) where {F} = IncompleteNormEstimator(F[], F(0.0))

function (convcrit::IncompleteNormEstimator{F})(
    rcbuffer::AbstractVector{K}, npivot::Int, tol::F
) where {F<:Real,K}
    isnotconverged = norm(rcbuffer) > tol * convcrit.normUV
    if !isnotconverged
        meany = mean(log10.(convcrit.lastnorms))
        x = Vector(1:(npivot - 1))
        meanx = npivot / 2

        β =
            sum((x .- meanx) .* (log10.(convcrit.lastnorms) .- meany)) /
            sum((x .- meanx) .^ 2)
        α = meany - β * meanx
        push!(convcrit.lastnorms, norm(rcbuffer))
        return (α + β * length(convcrit.lastnorms)) > log10(norm(rcbuffer)) #|| (α + β*(length(oldnorms)+1)) > log10(1e-4*oldnorms[1])
    else
        push!(convcrit.lastnorms, norm(rcbuffer))
        return isnotconverged
    end
end

function updatenorm!(
    convcrit::IncompleteNormEstimator{F}, rc::AbstractVector{K}, npivot::Int
) where {F<:Real,K}
    return convcrit.normUV = ((npivot - 1) * convcrit.normUV + norm(rc)) / npivot
end
