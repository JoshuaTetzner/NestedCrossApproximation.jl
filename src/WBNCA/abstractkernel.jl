struct AbstractKernel{K} <: AbstractMatrix{K}
    blockassembler::Function
end

function AbstractKernel(
    operator::BEAST.IntegralOperator, testspace::BEAST.Space, trialspace::BEAST.Space
)
    return AbstractKernel{scalartype(operator)}(
        BEAST.blockassembler(operator, testspace, trialspace)
    )
end

struct AbstractSubKernel{K} <: AbstractMatrix{K}
    rowidcs::Vector{Int}
    colidcs::Vector{Int}
    blockassembler::Function
end

function (M::AbstractSubKernel{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(M.rowidcs[i], M.colidcs[j], store)
end

function (M::AbstractKernel{K})(rowidcs::Vector{Int}, colidcs::Vector{Int}) where {K}
    return AbstractSubKernel(rowidcs, colidcs, M)
end

function (M::AbstractKernel{K})(
    buf::AbstractArray{K}, i::AbstractArray{Int,1}, j::AbstractArray{Int,1}
) where {K}
    @views store(v, m, n) = (buf[m, n] += v)
    return M.blockassembler(i, j, store)
end

AdaptiveCrossApproximation.nextrc!(buf, A::AbstractKernel, i, j) = A(buf, i, j)
AdaptiveCrossApproximation.nextrc!(buf, A::AbstractSubKernel, i, j) = A(buf, i, j)
