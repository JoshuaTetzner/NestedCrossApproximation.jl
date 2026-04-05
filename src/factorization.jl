import H2Trees: numberofvalues

#=
@inline _nworkers(::SerialScheduler) = 1
@inline _nworkers(scheduler::Any) = Threads.nthreads()
=#
@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.ACA, maxrank::Int
)
    return factorization([1], [1])
end

@inline function _stateful_factorization(
    factorization::AdaptiveCrossApproximation.iACA, maxrank::Int
)
    return factorization([1], [1], maxrank)
end
#=
@inline _stateful_factorization(factorization, maxrank::Int) = factorization

function factorization_channel(compressor; scheduler=DynamicScheduler(), maxrank::Int=40)
    nworkers = _nworkers(scheduler)
    prototype = _stateful_factorization(compressor.factorization, maxrank)
    channel = Channel{typeof(prototype)}(nworkers)
    for _ in 1:nworkers
        put!(channel, deepcopy(prototype))
    end
    return channel
end=#

#adapt_farfield_indices(factorization, representor, tree, Fidcs::Vector{Int}) =
#    isnothing(representor) ? Fidcs : representor(Fidcs)

function _ranklimit(maxrank::Int, nrows::Int, ncols::Int)
    return min(maxrank, min(nrows, ncols))
end

_use_tree_mimicry(f::AdaptiveCrossApproximation.iACA) =
    (f.rowpivoting isa AdaptiveCrossApproximation.TreeMimicryPivoting) ||
    (f.rowpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor) ||
    (f.columnpivoting isa AdaptiveCrossApproximation.TreeMimicryPivoting) ||
    (f.columnpivoting isa AdaptiveCrossApproximation.TreeMimicryPivotingFunctor)

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

#adapt_farfield_indices(compressor, tree, Fidcs::Vector{Int}) =
#    isnothing(compressor.representor) ? Fidcs : compressor.representor(Fidcs)

function adapt_farfield_indices(
    compressor::AdaptiveCrossApproximation.iACA, tree, Fidcs::Vector{Int}
)
    @assert allunique(Fidcs) "Expected unique far-field cluster ids (Fidcs)."
    if _use_tree_mimicry(compressor.factorization)
        return Fidcs
    end
    Fvalues = H2Trees.values(tree, Fidcs)
    return Fvalues#isnothing(compressor.representor) ? Fvalues : compressor.representor(Fvalues)
end

function adapt_farfield_indices(
    compressor::AdaptiveCrossApproximation.ACA, tree, Fidcs::Vector{Int}
)
    Fvalues = H2Trees.values(tree, Fidcs)
    #println("In $(length(Fidcs)) -> $(length(Fvalues))")
    return Fvalues#isnothing(compressor.representor) ? Fvalues : compressor.representor(Fvalues)
end

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
    Ftidcs = adapt_farfield_indices(factorization, trialtree(tree), Ftidcs)
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

    #=sidcs = H2Trees.values(H2Trees.trialtree(tree), Ftidcs)
    blk = zeros(eltype(farmatrix), length(tidcs), length(sidcs))
    farmatrix(blk, tidcs, sidcs)
    r = [findfirst(==(r), tidcs) for r in rows[1:npivots]]
    c = [findfirst(==(c), sidcs) for c in cols[1:npivots]]
    if norm(blk - blk[:, c] * inv(blk[r, c]) * blk[r, :]) / norm(blk) > 2e-1
        println(
            size(blk),
            "; ",
            "npivots: ",
            npivots,
            ", relerr: ",
            norm(blk - blk[:, c] * inv(blk[r, c]) * blk[r, :]) / norm(blk),
            ", \n",
            factorization.convergence.lastnorms[1:npivots],
        )
        println("r = ", tidcs)
        println("c = ", Ftidcs)
    end=#

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
    Fsidcs::Vector{Int},
    sidcs::Vector{Int},
    colbuffer::AbstractMatrix{T},
    rowbuffer::AbstractMatrix{T};
    maxrank::Int=40,
) where {T}
    Fsidcs = adapt_farfield_indices(factorization, testtree(tree), Fsidcs)

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
