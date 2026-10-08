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
    idxtodiridxptr = Vector{Int}(undef, nassigned + 1)
    idxtodiridxptr[1] = 1
    i = 1
    @inbounds for node in eachindex(cntr)
        iszero(cntr[node]) && continue
        nodes[i] = node
        idxtodiridxptr[i + 1] = idxtodiridxptr[i] + cntr[node]
        i += 1
    end

    return BasisStore{T}(DirBasisTraversalPlan(nodes, idxtodiridxptr, diridcs), blocks)
end

"""
    nestedtestbasis(tvalues, pivots, buffer)
    nestedtrialbasis(svalues, pivots, buffer)

Compute the basis of a leaf cluster from the pivot columns (rows) of the matrix stored in
`buffer`.

With the row pivots `τ` and the column pivots `σ` of a test cluster `t` the basis is
`A[t, σ] * inv(A[τ, σ])`, for a trial cluster `s` it is `inv(A[τ, σ]) * A[τ, s]`.

# Arguments

  - `tvalues`, `svalues`: indices of the cluster
  - `pivots`: pivots of the cluster, `τ` on the test and `σ` on the trial side
  - `buffer`: the first `length(pivots)` columns (rows) hold `A[:, σ]` (`A[τ, :]`)
"""
function nestedtestbasis(
    tvalues::Vector{Int}, pivots::Vector{Int}, buffer::AbstractMatrix{K}
) where {K}
    n = length(pivots)
    basis = buffer[tvalues, 1:n]
    return rdiv!(basis, _crossfactorization(buffer[pivots, 1:n], :nested_test_basis))
end

@doc (@doc nestedtestbasis) function nestedtrialbasis(
    svalues::Vector{Int}, pivots::Vector{Int}, buffer::AbstractMatrix{K}
) where {K}
    n = length(pivots)
    basis = buffer[1:n, svalues]
    return ldiv!(_crossfactorization(buffer[1:n, pivots], :nested_trial_basis), basis)
end
