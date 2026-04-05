function _build_basis_storage(bases::Vector{Matrix{T}}) where {T}
    blocks, nodes = collect_assigned(bases)
    return BasisStore{T}(BasisTraversalPlan(nodes), blocks)
end

function _build_dirbasis_storage(bases::Vector{Matrix{T}}, cntr::Vector{Int}) where {T}
    nassigned = 0
    @inbounds for nb in cntr
        nb != 0 && (nassigned += 1)
    end
    blocks, diridcs = collect_assigned(bases)

    nodes = Vector{Int}(undef, nassigned)
    diridxptr = Vector{Int}(undef, nassigned + 1)
    diridxptr[1] = 1
    i = 1
    @inbounds for node in eachindex(cntr)
        iszero(cntr[node]) && continue
        nodes[i] = node
        diridxptr[i + 1] = diridxptr[i] + cntr[node]
        i += 1
    end

    return BasisStore{T}(DirBasisTraversalPlan(nodes, diridxptr, diridcs), blocks)
end

function nestedtestbasis(
    tvalues::Vector{Int}, pivots::Vector{Int}, buffer::AbstractMatrix{K}
) where {K}
    return buffer[tvalues, 1:length(pivots)] / buffer[pivots, 1:length(pivots)]
end

function nestedtrialbasis(
    svalues::Vector{Int}, pivots::Vector{Int}, buffer::AbstractMatrix{K}
) where {K}
    return buffer[1:length(pivots), pivots] \ buffer[1:length(pivots), svalues]
end
