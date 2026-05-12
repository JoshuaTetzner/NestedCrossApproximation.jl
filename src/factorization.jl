import H2Trees: numberofvalues

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

@inline function _stateful_representor(representor::MimicryRep, maxrank::Int)
    return MimicryRepFunctor(representor.pivoting([1], [1], maxrank))
end

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

_effective_far_count(::AdaptiveCrossApproximation.ACA, tree, Fidcs::Vector{Int}) =
    length(Fidcs)

function _effective_far_count(
    factorization::AdaptiveCrossApproximation.iACA, tree, Fidcs::Vector{Int}
)
    return if _use_tree_mimicry(factorization)
        sum(numberofvalues(tree, node) for node in Fidcs)
    else
        length(Fidcs)
    end
end

function adapt_farfield_indices(
    factorization::CP, representor::RP, tree, t::Vector{Int}, Ft::Vector{Int}
) where {RP<:MimicryRepFunctor,CP}
    npivots = min(length(t), sum(numberofvalues(tree, node) for node in Ft))
    return representor(t, Ft, npivots)
end

adapt_farfield_indices(
    factorization::CP, ::Nothing, tree, t::Vector{Int}, Ft::Vector{Int}
) where {CP} = adapt_farfield_indices(factorization, tree, Ft)

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
    Ftidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank::Int=40,
) where {T}
    Ftidcs = adapt_farfield_indices(
        factorization, representor, trialtree(tree), tidcs, Ftidcs
    )
    ranklimit = _ranklimit(
        maxrank, length(tidcs), _effective_far_count(factorization, trialtree(tree), Ftidcs)
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

    #if Ftidcs == [214] && length(tidcs) == 41
    sidcs = H2Trees.values(H2Trees.trialtree(tree), Ftidcs)
    blk = zeros(eltype(farmatrix), length(tidcs), length(sidcs))
    farmatrix(blk, tidcs, sidcs)
    r = [findfirst(==(r), tidcs) for r in rows[1:npivots]]
    c = [findfirst(==(c), sidcs) for c in cols[1:npivots]]
    if norm(blk - blk[:, c] * inv(blk[r, c]) * blk[r, :]) / norm(blk) > 5e-3
        println(
            size(blk),
            "; ",
            "npivots: ",
            npivots,
            ", relerr: ",
            norm(blk - blk[:, c] * inv(blk[r, c]) * blk[r, :]) / norm(blk),
            "; ",
            tidcs,
            "; ",
            Ftidcs,
        )
        error()
    end
    npivots == maxrank && @warn "Maximum rank block"
    ##
    if npivots == maxrank
        println(npivots, ", ", maxrank)
        error()
    end
    ##
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
    Fs::Vector{Int},
    sidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank::Int=40,
) where {T}
    Fsidcs = adapt_farfield_indices(factorization, representor, testtree(tree), sidcs, Fs)
    ranklimit = _ranklimit(
        maxrank, _effective_far_count(factorization, testtree(tree), Fsidcs), length(sidcs)
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
    colbuffer[tidcs, 1:npivots] =
        colbuffer[tidcs, 1:npivots] * rowbuffer[1:npivots, cols[1:npivots]]
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
    colbuffer[tidcs, 1:npivots] =
        colbuffer[tidcs, 1:npivots] * rowbuffer[1:npivots, 1:npivots]
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
    rowbuffer[1:npivots, sidcs] =
        colbuffer[rows[1:npivots], 1:npivots] * rowbuffer[1:npivots, sidcs]
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
    rowbuffer[1:npivots, sidcs] =
        colbuffer[1:npivots, 1:npivots] * rowbuffer[1:npivots, sidcs]
    colbuffer[1:npivots, 1:npivots] .= 0
    return nothing
end
