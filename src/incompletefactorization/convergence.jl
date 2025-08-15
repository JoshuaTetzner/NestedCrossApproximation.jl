using Statistics, Polynomials, Test

mutable struct IncompleteNormEstimator{F} <: LRF.ConvCrit
    lastnorms::Vector{F}
    normUV::F
end

(::IncompleteNormEstimator{F})(kwargs...) where {F} = IncompleteNormEstimator(F[], F(0.0))

function (convcrit::IncompleteNormEstimator{F})(
    rcbuffer::AbstractVector{K}, npivot::Int, tol::F
) where {F<:Real,K}
    isnotconverged = norm(rcbuffer) > tol * convcrit.normUV

    if !isnotconverged && !isapprox(norm(rcbuffer), 0.0; atol=eps(real(eltype(rcbuffer))))
        y = log10.(convcrit.lastnorms)
        x = Vector(1:(npivot - 1))
        f2 = fit(x, y, 1)

        push!(convcrit.lastnorms, norm(rcbuffer))

        return f2(npivot) > log10(tol * convcrit.normUV)
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
