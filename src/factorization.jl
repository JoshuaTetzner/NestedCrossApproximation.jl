# Per-node point count without materializing the full point-id array (H2Trees.values
# recursively concatenates leaf data; counting only needs leaf sizes).
function _clustersize(tree, node::Int)
    isleaf(tree, node) && return length(H2Trees.values(H2Trees.data(tree, node)))
    n = 0
    for leaf in H2Trees.leaves(tree, node)
        n += length(H2Trees.values(H2Trees.data(tree, Int(leaf))))
    end
    return n
end

# Marks far-field indices that are already point-level. TopDown must merge its own
# far-field node's points with inherited point-level parent pivots before calling
# compute_test_pivots!/compute_trial_pivots! (BottomUp never needs this — it always
# passes raw node ids). Wrapping that merged list tells adapt_farfield_indices to
# skip the node->point conversion it would otherwise apply.
struct PreResolvedIndices
    idcs::Vector{Int}
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.ACA, maxrank::Int
)
    return factorization([1], [1])
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.ACA, A::AbstractKernelMatrix, maxrank::Int
)
    return factorization(A, [1], [1], maxrank)
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.iACA, maxrank::Int
)
    return factorization([1], [1], maxrank)
end

# ACA's 2-arg form only resolves for the literal default MaximumValue/MaximumValue/
# FNormEstimator combo, and even then builds the convergence criterion without
# seeing the matrix or maxrank. The 3-arg form is the generically-correct one, so
# always route ACA through it; iACA only has the maxrank-aware 2-arg form.
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

# RandomRep/ChebyshevRep are stateless and already implement the (refidcs, idcs, npivots)
# call protocol directly, so they need no per-task conversion.
@inline _stateful_representor(representor::Representor, maxrank::Int) = representor

@inline function _stateful_representor(::Nothing, maxrank::Int)
    return nothing
end

function _ranklimit(maxrank::Int, nrows::Int, ncols::Int)
    return min(maxrank, min(nrows, ncols))
end

_use_tree_mimicry(f::AdaptiveCrossApproximation.iACA) =
    (f.rowpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor) ||
    (f.rowpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor2) ||
    (f.columnpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor) ||
    (f.columnpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor2)

# A representor has already reduced/capped Fidcs to the achievable representative count,
# so its resolved length is authoritative regardless of factorization type or Ft's shape.
_effective_far_count(
    factorization, tree, Ft, representor::Representor, Fidcs::Vector{Int}
) = length(Fidcs)

# Pre-resolved Ft is always point-level by construction, regardless of factorization
# type, so its effective count is always just its length (bypasses the tree-mimicry
# per-cluster counting below, which only makes sense for genuinely node-level Ft).
_effective_far_count(
    factorization, tree, ::PreResolvedIndices, ::Nothing, Fidcs::Vector{Int}
) = length(Fidcs)

_effective_far_count(
    ::AdaptiveCrossApproximation.ACA, tree, ::Vector{Int}, ::Nothing, Fidcs::Vector{Int}
) = length(Fidcs)

function _effective_far_count(
    factorization::AdaptiveCrossApproximation.iACA,
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

# MimicryRep needs node-level Ft so it can walk the tree top-down from a coherent cluster.
# PreResolvedIndices is already a flattened, multi-origin point set (TopDown's own
# far-field node merged with inherited parent pivots) with no single node to walk from,
# so it isn't supported there - only with BottomUp, which always passes raw node ids.
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

# RandomRep/ChebyshevRep operate on flat point indices regardless of origin, so they work
# uniformly for raw node-level Ft (BottomUp) and pre-resolved point-level Ft (TopDown).
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
    factorization::AdaptiveCrossApproximation.iACA, tree, Ft::Vector{Int}
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
    ::AdaptiveCrossApproximation.iACA,
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
    ::AdaptiveCrossApproximation.iACA,
    colbuffer::AbstractMatrix,
    Fsidcs::Vector{Int},
    ranklimit::Int,
) = colbuffer

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
    npivots == maxrank && @warn "Maximum rank block"

    _finalize_test_buffers!(
        factorization, colbuffer, rowbuffer, tidcs, Ftidcs, rows, cols, npivots
    )

    return rows[1:npivots], cols[1:npivots]
end

function compute_trial_pivots!(
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
    npivots == maxrank && @warn "Maximum rank block"
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
    factorization::AdaptiveCrossApproximation.iACA,
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
    rowbuffer[1:npivots, Ftidcs] .= 0
    return nothing
end

function _finalize_test_buffers!(
    ::AdaptiveCrossApproximation.iACA,
    colbuffer,
    rowbuffer,
    tidcs,
    Ftidcs,
    rows,
    cols,
    npivots,
)
    _right_multiply_rows!(colbuffer, tidcs, rowbuffer, npivots)
    rowbuffer[1:npivots, 1:npivots] .= 0
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
    colbuffer[Fsidcs, 1:npivots] .= 0
    return nothing
end

function _finalize_trial_buffers!(
    ::AdaptiveCrossApproximation.iACA,
    colbuffer,
    rowbuffer,
    Fsidcs,
    sidcs,
    rows,
    cols,
    npivots,
)
    _left_multiply_cols!(colbuffer, rowbuffer, sidcs, npivots)
    colbuffer[1:npivots, 1:npivots] .= 0
    return nothing
end
