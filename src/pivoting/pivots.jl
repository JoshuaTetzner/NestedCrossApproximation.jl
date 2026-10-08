# Number of points of a node, without collecting them.
function _clustersize(tree, node::Int)
    isleaf(tree, node) && return length(H2Trees.values(H2Trees.data(tree, node)))
    n = 0
    for leaf in H2Trees.leaves(tree, node)
        n += length(H2Trees.values(H2Trees.data(tree, Int(leaf))))
    end
    return n
end

# Far-field indices that are points already instead of nodes: TopDown merges the points
# of a cluster's far nodes with the far pivots of its parent.
struct PreResolvedIndices
    idcs::Vector{Int}
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.ACA, A::AbstractKernelMatrix, maxrank::Int
)
    return factorization(A, [1], [1], maxrank)
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.IACA, maxrank::Int
)
    return factorization([1], [1], maxrank)
end

# ACA builds its convergence criterion from the matrix and maxrank, IACA from maxrank.
@inline function _build_stateful_factorization(
    factorization::AdaptiveCrossApproximation.ACA, farmatrix, maxrank::Int
)
    return _stateful_factorization(factorization, farmatrix, maxrank)
end

@inline function _build_stateful_factorization(factorization, farmatrix, maxrank::Int)
    return _stateful_factorization(factorization, maxrank)
end

@inline function _stateful_representor(representor::MimicryRep, maxrank::Int)
    return MimicryRepFunctor(representor.pivoting([1], [1], maxrank))
end

@inline _stateful_representor(representor::Representor, maxrank::Int) = representor

@inline function _stateful_representor(::Nothing, maxrank::Int)
    return nothing
end

function _ranklimit(maxrank::Int, nrows::Int, ncols::Int)
    return min(maxrank, min(nrows, ncols))
end

_use_tree_mimicry(f::AdaptiveCrossApproximation.IACA) =
    (f.rowpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor) ||
    (f.columnpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor)

# A representor has already reduced the far field to points.
_effective_far_count(
    factorization, tree, Ft, representor::Representor, Fidcs::Vector{Int}
) = length(Fidcs)

_effective_far_count(
    factorization, tree, ::PreResolvedIndices, ::Nothing, Fidcs::Vector{Int}
) = length(Fidcs)

_effective_far_count(
    ::AdaptiveCrossApproximation.ACA, tree, ::Vector{Int}, ::Nothing, Fidcs::Vector{Int}
) = length(Fidcs)

function _effective_far_count(
    factorization::AdaptiveCrossApproximation.IACA,
    tree,
    ::Vector{Int},
    ::Nothing,
    Fidcs::Vector{Int},
)
    return if _use_tree_mimicry(factorization)
        sum(_clustersize(tree, node) for node in Fidcs)
    else
        length(Fidcs)
    end
end

adapt_farfield_indices(
    factorization, ::Nothing, tree, t::Vector{Int}, Ft::PreResolvedIndices
) = Ft.idcs

# MimicryRep descends the tree from the far nodes, which a point-level far field lacks.
function adapt_farfield_indices(
    factorization, ::MimicryRepFunctor, tree, t::Vector{Int}, Ft::PreResolvedIndices
)
    return throw(
        ArgumentError(
            "MimicryRep requires node-level far-field indices and is only supported " *
            "with BottomUp; got pre-resolved point-level indices (TopDown).",
        ),
    )
end

function adapt_farfield_indices(
    factorization, representor::MimicryRepFunctor, tree, t::Vector{Int}, Ft::Vector{Int}
)
    npivots = min(length(t), sum(_clustersize(tree, node) for node in Ft))
    return representor(t, Ft, npivots)
end

function adapt_farfield_indices(
    factorization, representor::Representor, tree, t::Vector{Int}, Ft::PreResolvedIndices
)
    npivots = min(length(t), length(Ft.idcs))
    return representor(t, Ft.idcs, npivots)
end

function adapt_farfield_indices(
    factorization, representor::Representor, tree, t::Vector{Int}, Ft::Vector{Int}
)
    Fvalues = H2Trees.values(tree, Ft)
    npivots = min(length(t), length(Fvalues))
    return representor(t, Fvalues, npivots)
end

adapt_farfield_indices(factorization, ::Nothing, tree, t::Vector{Int}, Ft::Vector{Int}) =
    adapt_farfield_indices(factorization, tree, Ft)

function adapt_farfield_indices(
    factorization::AdaptiveCrossApproximation.IACA, tree, Ft::Vector{Int}
)
    @assert allunique(Ft) "Expected unique far-field cluster ids (Ft)."
    if _use_tree_mimicry(factorization)
        return Ft
    end
    Fvalues = H2Trees.values(tree, Ft)
    return Fvalues
end

adapt_farfield_indices(::AdaptiveCrossApproximation.ACA, tree, Ft::Vector{Int}) =
    H2Trees.values(tree, Ft)

_test_rowbuffer_for_rawpivots(
    ::AdaptiveCrossApproximation.ACA,
    rowbuffer::AbstractMatrix,
    Ftidcs::Vector{Int},
    ranklimit::Int,
) = view(rowbuffer, 1:ranklimit, Ftidcs)

_test_rowbuffer_for_rawpivots(
    ::AdaptiveCrossApproximation.IACA,
    rowbuffer::AbstractMatrix,
    Ftidcs::Vector{Int},
    ranklimit::Int,
) = rowbuffer

_trial_colbuffer_for_rawpivots(
    ::AdaptiveCrossApproximation.ACA,
    colbuffer::AbstractMatrix,
    Fsidcs::Vector{Int},
    ranklimit::Int,
) = view(colbuffer, Fsidcs, 1:ranklimit)

_trial_colbuffer_for_rawpivots(
    ::AdaptiveCrossApproximation.IACA,
    colbuffer::AbstractMatrix,
    Fsidcs::Vector{Int},
    ranklimit::Int,
) = colbuffer

"""
    compute_test_pivots!(factorization, representor, farmatrix, tree, tidcs, Ft,
        colbuffer, rowbuffer; maxrank=40)
    compute_trial_pivots!(factorization, representor, farmatrix, tree, Fs, sidcs,
        colbuffer, rowbuffer; maxrank=40)

Select the pivots of a cluster by a low-rank factorization of the block of the cluster
against its far field.

# Arguments

  - `factorization`: `ACA` or `IACA` prepared for the block
  - `representor`: [`Representor`](@ref) that reduces the far field, or `nothing`
  - `farmatrix`: kernel matrix for the evaluation of far-field entries
  - `tree`: block tree
  - `tidcs`, `sidcs`: candidates for the pivots of the cluster
  - `Ft`, `Fs`: far field of the cluster as nodes of the other tree, or as points wrapped
    in `PreResolvedIndices`
  - `colbuffer`, `rowbuffer`: buffers of the factorization

# Returns

  - `rows`, `cols`: row and column pivots. On the test side `rows` are the pivots of the
    cluster and `cols` those in the far field, on the trial side vice versa.

# Notes

On return the buffer of the cluster side (`colbuffer` on the test side) holds the pivot
columns (rows) of the cluster as required by [`nestedtestbasis`](@ref) and
[`testtransfer`](@ref); the other buffer is reset to zero.
"""
function compute_test_pivots!(
    factorization,
    representor,
    farmatrix::AbstractKernelMatrix{T},
    tree::H2Trees.BlockTree,
    tidcs::Vector{Int},
    Ft::Union{Vector{Int},PreResolvedIndices},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank::Int=40,
) where {T}
    Ftidcs = adapt_farfield_indices(factorization, representor, trialtree(tree), tidcs, Ft)
    ranklimit = _ranklimit(
        maxrank,
        length(tidcs),
        _effective_far_count(factorization, trialtree(tree), Ft, representor, Ftidcs),
    )

    colbuffer[tidcs, 1:ranklimit] .= 0
    rawrowbuffer = _test_rowbuffer_for_rawpivots(
        factorization, rowbuffer, Ftidcs, ranklimit
    )
    rows = Vector{Int}(undef, ranklimit)
    cols = Vector{Int}(undef, ranklimit)
    npivots, rows, cols = _compute_raw_pivots!(
        factorization,
        farmatrix,
        tidcs,
        Ftidcs,
        view(colbuffer, tidcs, 1:ranklimit),
        rawrowbuffer,
        rows,
        cols,
        ranklimit,
    )
    npivots == maxrank && @warn "Maximum rank block" maxlog = 1

    _finalize_test_buffers!(
        factorization, colbuffer, rowbuffer, tidcs, Ftidcs, rows, cols, npivots
    )

    return rows[1:npivots], cols[1:npivots]
end

@doc (@doc compute_test_pivots!) function compute_trial_pivots!(
    factorization,
    representor,
    farmatrix::AbstractKernelMatrix{T},
    tree::H2Trees.BlockTree,
    Fs::Union{Vector{Int},PreResolvedIndices},
    sidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank::Int=40,
) where {T}
    Fsidcs = adapt_farfield_indices(factorization, representor, testtree(tree), sidcs, Fs)
    ranklimit = _ranklimit(
        maxrank,
        _effective_far_count(factorization, testtree(tree), Fs, representor, Fsidcs),
        length(sidcs),
    )

    rowbuffer[1:ranklimit, sidcs] .= 0
    rawcolbuffer = _trial_colbuffer_for_rawpivots(
        factorization, colbuffer, Fsidcs, ranklimit
    )
    rows = Vector{Int}(undef, ranklimit)
    cols = Vector{Int}(undef, ranklimit)
    npivots, rows, cols = _compute_raw_pivots!(
        factorization,
        farmatrix,
        Fsidcs,
        sidcs,
        rawcolbuffer,
        view(rowbuffer, 1:ranklimit, sidcs),
        rows,
        cols,
        ranklimit,
    )
    npivots == maxrank && @warn "Maximum rank block" maxlog = 1
    _finalize_trial_buffers!(
        factorization, colbuffer, rowbuffer, Fsidcs, sidcs, rows, cols, npivots
    )

    return rows[1:npivots], cols[1:npivots]
end

function _compute_raw_pivots!(
    factorization::AdaptiveCrossApproximation.ACA,
    farmatrix,
    rowidcs,
    colidcs,
    colbuffer,
    rowbuffer,
    rowpivs,
    colpivs,
    ranklimit,
)
    AdaptiveCrossApproximation.reset!(factorization, rowidcs, colidcs)
    npivots = factorization(
        farmatrix, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, ranklimit
    )
    return npivots, rowpivs, colpivs
end

function _compute_raw_pivots!(
    factorization::AdaptiveCrossApproximation.IACA,
    farmatrix,
    rowidcs,
    colidcs,
    colbuffer,
    rowbuffer,
    rowpivs,
    colpivs,
    ranklimit,
)
    npivots, rowpivs, colpivs = factorization(
        farmatrix, colbuffer, rowbuffer, rowpivs, colpivs, rowidcs, colidcs, ranklimit
    )
    return npivots, rowpivs, colpivs
end

function _right_multiply_rows!(
    A::AbstractMatrix{T},
    rowidcs::AbstractVector{Int},
    right::AbstractMatrix,
    rightcols::AbstractVector{Int},
    n::Int,
) where {T}
    scratch = Vector{T}(undef, n)
    @inbounds for row in rowidcs
        for j in 1:n
            acc = zero(T)
            for k in 1:n
                acc += A[row, k] * right[k, rightcols[j]]
            end
            scratch[j] = acc
        end
        for j in 1:n
            A[row, j] = scratch[j]
        end
    end
    return A
end

function _right_multiply_rows!(
    A::AbstractMatrix{T}, rowidcs::AbstractVector{Int}, right::AbstractMatrix, n::Int
) where {T}
    scratch = Vector{T}(undef, n)
    @inbounds for row in rowidcs
        for j in 1:n
            acc = zero(T)
            for k in 1:n
                acc += A[row, k] * right[k, j]
            end
            scratch[j] = acc
        end
        for j in 1:n
            A[row, j] = scratch[j]
        end
    end
    return A
end

function _left_multiply_cols!(
    left::AbstractMatrix,
    leftrows::AbstractVector{Int},
    A::AbstractMatrix{T},
    colidcs::AbstractVector{Int},
    n::Int,
) where {T}
    scratch = Vector{T}(undef, n)
    @inbounds for col in colidcs
        for i in 1:n
            acc = zero(T)
            for k in 1:n
                acc += left[leftrows[i], k] * A[k, col]
            end
            scratch[i] = acc
        end
        for i in 1:n
            A[i, col] = scratch[i]
        end
    end
    return A
end

function _left_multiply_cols!(
    left::AbstractMatrix, A::AbstractMatrix{T}, colidcs::AbstractVector{Int}, n::Int
) where {T}
    scratch = Vector{T}(undef, n)
    @inbounds for col in colidcs
        for i in 1:n
            acc = zero(T)
            for k in 1:n
                acc += left[i, k] * A[k, col]
            end
            scratch[i] = acc
        end
        for i in 1:n
            A[i, col] = scratch[i]
        end
    end
    return A
end

function _finalize_test_buffers!(
    ::AdaptiveCrossApproximation.ACA,
    colbuffer,
    rowbuffer,
    tidcs,
    Ftidcs,
    rows,
    cols,
    npivots,
)
    _right_multiply_rows!(colbuffer, tidcs, rowbuffer, cols, npivots)
    # One row more: a pivot the search rejected is still in the buffer.
    rowbuffer[1:min(npivots + 1, size(rowbuffer, 1)), Ftidcs] .= 0
    return nothing
end

function _finalize_test_buffers!(
    ::AdaptiveCrossApproximation.IACA,
    colbuffer,
    rowbuffer,
    tidcs,
    Ftidcs,
    rows,
    cols,
    npivots,
)
    _right_multiply_rows!(colbuffer, tidcs, rowbuffer, npivots)
    nrows = min(npivots + 1, size(rowbuffer, 1))
    ncols = min(npivots + 1, size(rowbuffer, 2))
    rowbuffer[1:nrows, 1:ncols] .= 0
    return nothing
end

function _finalize_trial_buffers!(
    ::AdaptiveCrossApproximation.ACA,
    colbuffer,
    rowbuffer,
    Fsidcs,
    sidcs,
    rows,
    cols,
    npivots,
)
    _left_multiply_cols!(colbuffer, rows, rowbuffer, sidcs, npivots)
    colbuffer[Fsidcs, 1:min(npivots + 1, size(colbuffer, 2))] .= 0
    return nothing
end

function _finalize_trial_buffers!(
    ::AdaptiveCrossApproximation.IACA,
    colbuffer,
    rowbuffer,
    Fsidcs,
    sidcs,
    rows,
    cols,
    npivots,
)
    _left_multiply_cols!(colbuffer, rowbuffer, sidcs, npivots)
    nrows = min(npivots + 1, size(colbuffer, 1))
    ncols = min(npivots + 1, size(colbuffer, 2))
    colbuffer[1:nrows, 1:ncols] .= 0
    return nothing
end
