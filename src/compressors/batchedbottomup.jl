import H2Trees: levels, LevelIterator

"""
    BatchedBottomUp{PivotSelectorType,CompressorType}

Compressor that constructs the nested bases from the leaves to the root and evaluates
the matrix entries of a whole level in one batch.

The construction has three phases:

 1. The far-field pivots of every cluster are selected from the geometry alone, without
    matrix entries.
 2. For each cluster of a level one dense block against its far-field pivots is evaluated.
    Its other indices are all points of a leaf, or the pivots of the children.
 3. The pivots of the cluster are selected in this block, and its basis or transfer
    matrices are computed from it.

# Fields

  - `pivotselector::PivotSelectorType`: geometric pivoting strategy for the first phase,
    e.g. `AdaptiveCrossApproximation.TreeMimicryPivoting`
  - `compressor::CompressorType`: factorization of a dense block for the third phase, e.g.
    a fully pivoted `ACA`

# Notes

The batched evaluation leaves the choice between CPU and GPU to the kernel matrix.
Directional H²-matrices require [`BottomUp`](@ref).
"""
mutable struct BatchedBottomUp{PivotSelectorType,CompressorType}
    pivotselector::PivotSelectorType
    compressor::CompressorType

    function BatchedBottomUp(pivotselector, compressor)
        return new{typeof(pivotselector),typeof(compressor)}(pivotselector, compressor)
    end
end

"""
    BatchedBottomUp(; pivotselector, compressor)

Build a [`BatchedBottomUp`](@ref) compressor from keyword arguments.

# Keyword Arguments

  - `pivotselector`: geometric pivoting strategy that selects the far-field pivots
  - `compressor`: factorization of a dense block
"""
function BatchedBottomUp(; pivotselector, compressor)
    return BatchedBottomUp(pivotselector, compressor)
end

# Phase 1 for one node: up to `maxrank` points out of the far nodes `idcs`.
function _selectfarpivots(
    pivotselector, refidcs::Vector{Int}, idcs::Vector{Int}, fartree, maxrank::Int
)
    limit = min(maxrank, length(refidcs))
    n = 0
    for node in idcs
        n += _clustersize(fartree, node, limit - n)
        n == limit && break
    end
    n == 0 && return Int[]

    functor = pivotselector(refidcs, idcs, n)
    pivots = Vector{Int}(undef, n)
    pivots[1] = functor()
    for npivot in 2:n
        pivots[npivot] = functor(npivot)
    end
    return pivots
end

# Opt-in per-level stage timing (assemble vs. compress), matching the
# `ACA_GPU_TIMING` convention on the assembly side. Enable with
# `NCA_BATCHED_TIMING=1` to isolate the ACA+basis/transfer share against the
# block-assembly share, on either the CPU or GPU backend.
_batched_timing() = get(ENV, "NCA_BATCHED_TIMING", "0") == "1"

# ---------------------------------------------------------------------------
# Backend seams (Phase 2 + Phase 3). `testbases`/`trialbases` build the level
# metadata (Phase 1 book-keeping) and then call these two hooks; the CPU
# methods below hold the original code verbatim, while a GPU `farmatrix`
# supplies its own methods (NestedCrossApproximationCUDAExt) that keep the whole
# level resident on the device. `assemble_level` returns an opaque `storage`
# whose type selects which `compresslevel_*!` method runs, so the leaf->root
# traversal skeleton stays backend-agnostic and unduplicated.
# ---------------------------------------------------------------------------

"""
    assemble_level(farmatrix, rowidcs, colidcs; maxrank, ntasks, transposed=false) -> storage

Phase 2: assemble one tree level's dense far blocks (block `i` is
`rowidcs[i] × colidcs[i]`). The CPU method returns a plain `Vector{Matrix}`;
the returned storage type is the backend seam consumed by `compresslevel_*!`.
`maxrank` is unused here on CPU (blocks are exactly sized) but sets the padded
buffer width for the GPU method.
With `transposed=true`, evaluates `transpose(farmatrix[colidcs[i], rowidcs[i]])`
into the same layout, for trial-side compression without conjugation.
"""
function assemble_level(
    farmatrix::AbstractKernelMatrix{T},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    maxrank::Int,
    ntasks::Integer=1,
    transposed::Bool=false,
) where {T}
    blocks = [zeros(T, length(rowidcs[i]), length(colidcs[i])) for i in eachindex(rowidcs)]
    if transposed
        assemble_blocks(farmatrix, transpose.(blocks), colidcs, rowidcs; ntasks=ntasks)
    else
        assemble_blocks(farmatrix, blocks, rowidcs, colidcs; ntasks=ntasks)
    end
    return blocks
end

# Other dense compressors keep their original interface; full-pivoted ACA
# rescales its sampled residual in the large-far-field limit.
_compressblock(cfg, block, maxrank) = cfg(block, maxrank)
function _compressblock(
    cfg::ACA{RP,CP,C}, block, maxrank
) where {RP<:FullPivoting,CP<:FullPivoting,C<:FNormEstimator}
    return cfg(block, maxrank; sampled=true)
end

"""
    compresslevel_test!(cfg, storage, subtree, activeclusters, tvalues,
                        childranges, nodechildren, tpivots, bases, transfer,
                        maxrank, pbar; scheduler)

Phase 3 (test side): for each active cluster select its own pivots via the ACA
callable `cfg`, then build its leaf basis (`nestedtestbasis`) or child transfer
matrices (`testtransfer`) from the assembled block. Writes `tpivots`, `bases`,
`transfer` by global node id. Dispatched on `storage` (host `Vector{Matrix}`
here); the GPU method batches the same steps over a device stack.
Full-pivoted ACA rescales the squared sampled residual by `s/(s-r)` without
counting the full far-field columns.

The per-cluster loop runs under `scheduler` (`@tasks`), since each iteration
compresses an independent block and writes a distinct node id — the only shared
mutable state is the ACA scratch buffer, which is task-local (`@local`). The GPU
method ignores `scheduler` (a whole level is one batched launch).
"""
function compresslevel_test!(
    cfg,
    storage::AbstractVector{<:AbstractMatrix{T}},
    subtree,
    activeclusters,
    tvalues,
    childranges,
    nodechildren,
    tpivots,
    bases,
    transfer,
    maxrank::Int,
    pbar;
    scheduler=SerialScheduler(),
) where {T}
    maxrows = maximum(length, tvalues)
    maxcols = maximum(size(b, 2) for b in storage)
    @tasks for i in eachindex(activeclusters)
        @set scheduler = scheduler
        @local scratch = Matrix{T}(undef, maxrows, maxcols)

        t = activeclusters[i]
        m, n = size(storage[i])
        Ascratch = view(scratch, 1:m, 1:n)
        copyto!(Ascratch, storage[i])
        _, rowpivots, colpivots = _compressblock(cfg, Ascratch, maxrank)

        tpivots[t] = tvalues[i][rowpivots]
        localbuffer = storage[i][:, colpivots]

        if isleaf(subtree, t)
            bases[t] = nestedtestbasis(collect(1:m), rowpivots, localbuffer)
        else
            children = nodechildren[i]
            nodetransfers = Vector{Matrix{T}}(undef, length(children))
            for j in eachindex(children)
                nodetransfers[j] = testtransfer(rowpivots, childranges[i][j], localbuffer)
            end
            transfer[t] = nodetransfers
        end
        next!(pbar)
    end
    return nothing
end

"""
    compresslevel_trial!(cfg, storage, subtree, activeclusters, svalues,
                         childranges, nodechildren, spivots, bases, transfer,
                         maxrank, pbar)

Phase 3 (trial side): mirrors [`compresslevel_test!`], but
`nestedtrialbasis`/`trialtransfer` expect a buffer shaped
(far-pivot-rank × own-trial-dofs), the transpose of the block's own
orientation, so the local slice is transposed before use. `transpose`, not
adjoint: `T` is typically complex (BEM) and adjoint would wrongly conjugate.
"""
function compresslevel_trial!(
    cfg,
    storage::AbstractVector{<:AbstractMatrix{T}},
    subtree,
    activeclusters,
    svalues,
    childranges,
    nodechildren,
    spivots,
    bases,
    transfer,
    maxrank::Int,
    pbar;
    scheduler=SerialScheduler(),
) where {T}
    maxrows = maximum(length, svalues)
    maxcols = maximum(size(b, 2) for b in storage)
    @tasks for i in eachindex(activeclusters)
        @set scheduler = scheduler
        @local scratch = Matrix{T}(undef, maxrows, maxcols)

        s = activeclusters[i]
        m, n = size(storage[i])
        Ascratch = view(scratch, 1:m, 1:n)
        copyto!(Ascratch, storage[i])
        _, rowpivots, colpivots = _compressblock(cfg, Ascratch, maxrank)

        spivots[s] = svalues[i][rowpivots]
        localbuffer = Matrix(transpose(storage[i][:, colpivots]))

        if isleaf(subtree, s)
            bases[s] = nestedtrialbasis(collect(1:m), rowpivots, localbuffer)
        else
            children = nodechildren[i]
            nodetransfers = Vector{Matrix{T}}(undef, length(children))
            for j in eachindex(children)
                nodetransfers[j] = trialtransfer(rowpivots, childranges[i][j], localbuffer)
            end
            transfer[s] = nodetransfers
        end
        next!(pbar)
    end
    return nothing
end

function testbases(
    compressor::BatchedBottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    ntasks::Integer=Threads.nthreads(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    scheduler = _scheduler(ntasks)
    farpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    @tasks for t in eachindex(farpivots)
        @set scheduler = scheduler
        Ft = farfield(testtree(tree), fardata, t)
        if !isempty(Ft)
            tvalues = H2Trees.values(testtree(tree), t)
            farpivots[t] = _selectfarpivots(
                compressor.pivotselector, tvalues, Ft, trialtree(tree), maxrank
            )
        end
    end

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        activeclusters = filter(
            t -> isassigned(farpivots, t) && !isempty(farpivots[t]), testclusters
        )
        isempty(activeclusters) && continue

        # Rows of each block: the points of a leaf, the pivots of the children otherwise.
        # `childranges[i][j]` is the range of child j in `tvalues[i]`.
        tvalues = Vector{Vector{Int}}(undef, length(activeclusters))
        childranges = Vector{Vector{UnitRange{Int}}}(undef, length(activeclusters))
        nodechildren = Vector{Vector{Int}}(undef, length(activeclusters))
        for (i, t) in enumerate(activeclusters)
            if isleaf(testtree(tree), t)
                tvalues[i] = H2Trees.values(testtree(tree), t)
                childranges[i] = UnitRange{Int}[]
                nodechildren[i] = Int[]
            else
                children = collect(H2Trees.ChildIterator(testtree(tree), t))
                vals = Int[]
                ranges = Vector{UnitRange{Int}}(undef, length(children))
                for (j, child) in enumerate(children)
                    start = length(vals) + 1
                    append!(vals, tpivots[child])
                    ranges[j] = start:length(vals)
                end
                tvalues[i] = vals
                childranges[i] = ranges
                nodechildren[i] = children
            end
        end
        blockpivots = [farpivots[t] for t in activeclusters]

        # Phase 2 — one block per active node (own tvalues × its farpivots).
        t_assemble = @elapsed storage = assemble_level(
            farmatrix, tvalues, blockpivots; maxrank=maxrank, ntasks=ntasks
        )

        # Phase 3 — select pivots (batched ACA), then build the leaf basis or
        # transfer matrices straight from the assembled level.
        t_compress = @elapsed compresslevel_test!(
            compressor.compressor,
            storage,
            testtree(tree),
            activeclusters,
            tvalues,
            childranges,
            nodechildren,
            tpivots,
            bases,
            transfer,
            maxrank,
            pbar;
            scheduler=scheduler,
        )
        _batched_timing() && println(
            "[batchedbottomup test] level $level: $(length(activeclusters)) blocks | ",
            "assemble=$(round(t_assemble; digits=4))s | ",
            "compress(aca+basis/transfer)=$(round(t_compress; digits=4))s",
        )

        sizehint!(level_transfer_nodes[level], length(testclusters))
        @inbounds for t in testclusters
            if !isleaf(testtree(tree), t) && isassigned(transfer, t)
                push!(level_transfer_nodes[level], t)
            end
        end
    end

    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(transfer, level_transfer_nodes, testtree(tree))
    return basis_store, transfer_store, tpivots
end

function trialbases(
    compressor::BatchedBottomUp,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    ntasks::Integer=Threads.nthreads(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    scheduler = _scheduler(ntasks)
    farpivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    @tasks for s in eachindex(farpivots)
        @set scheduler = scheduler
        Fs = farfield(trialtree(tree), fardata, s)
        if !isempty(Fs)
            svalues = H2Trees.values(trialtree(tree), s)
            farpivots[s] = _selectfarpivots(
                compressor.pivotselector, svalues, Fs, testtree(tree), maxrank
            )
        end
    end

    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        activeclusters = filter(
            s -> isassigned(farpivots, s) && !isempty(farpivots[s]), trialclusters
        )
        isempty(activeclusters) && continue

        # Columns of each block, as the rows in testbases.
        svalues = Vector{Vector{Int}}(undef, length(activeclusters))
        childranges = Vector{Vector{UnitRange{Int}}}(undef, length(activeclusters))
        nodechildren = Vector{Vector{Int}}(undef, length(activeclusters))
        for (i, s) in enumerate(activeclusters)
            if isleaf(trialtree(tree), s)
                svalues[i] = H2Trees.values(trialtree(tree), s)
                childranges[i] = UnitRange{Int}[]
                nodechildren[i] = Int[]
            else
                children = collect(H2Trees.ChildIterator(trialtree(tree), s))
                vals = Int[]
                ranges = Vector{UnitRange{Int}}(undef, length(children))
                for (j, child) in enumerate(children)
                    start = length(vals) + 1
                    append!(vals, spivots[child])
                    ranges[j] = start:length(vals)
                end
                svalues[i] = vals
                childranges[i] = ranges
                nodechildren[i] = children
            end
        end
        blockpivots = [farpivots[s] for s in activeclusters]

        # Evaluate K[far test pivots, own trial values] through transposed
        # output views, keeping own trial values on rows for compression.
        t_assemble = @elapsed storage = assemble_level(
            farmatrix, svalues, blockpivots; maxrank=maxrank, ntasks=ntasks, transposed=true
        )

        # Compress trial-index rows, then store factors in trial orientation.
        t_compress = @elapsed compresslevel_trial!(
            compressor.compressor,
            storage,
            trialtree(tree),
            activeclusters,
            svalues,
            childranges,
            nodechildren,
            spivots,
            bases,
            transfer,
            maxrank,
            pbar;
            scheduler=scheduler,
        )
        _batched_timing() && println(
            "[batchedbottomup trial] level $level: $(length(activeclusters)) blocks | ",
            "assemble=$(round(t_assemble; digits=4))s | ",
            "compress(aca+basis/transfer)=$(round(t_compress; digits=4))s",
        )

        sizehint!(level_transfer_nodes[level], length(trialclusters))
        @inbounds for s in trialclusters
            if !isleaf(trialtree(tree), s) && isassigned(transfer, s)
                push!(level_transfer_nodes[level], s)
            end
        end
    end

    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(
        transfer, level_transfer_nodes, trialtree(tree)
    )
    return basis_store, transfer_store, spivots
end
