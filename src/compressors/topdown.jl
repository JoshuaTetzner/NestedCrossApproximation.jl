import H2Trees: levels, LevelIterator

"""
    TopDown{LowRankFactorizationType,RepresentorType}

Compressor that constructs the nested bases from the root to the leaves.

For each cluster the block of the cluster against its far field is factorized. The far
field consists of the points of the admissible clusters and of the far-field pivots of the
parent, through which the far field of all ancestors is taken into account. The selected
pivots of the cluster define its basis and transfer matrices.

# Fields

  - `factorization::LowRankFactorizationType`: low-rank factorization that selects the
    pivots, an `ACA` or an `IACA` of AdaptiveCrossApproximation.jl
  - `representor::RepresentorType`: [`Representor`](@ref) that reduces the far field before
    the factorization, or `nothing`

# Notes

The far field is a set of points. Hence `TreeMimicryPivoting` in an `IACA` and
[`MimicryRep`](@ref) are not supported; use `MimicryPivoting` instead. Directional
H²-matrices require [`BottomUp`](@ref).
"""
mutable struct TopDown{LowRankFactorizationType,RepresentorType}
    factorization::LowRankFactorizationType
    representor::RepresentorType

    function TopDown(factorization, representor)
        return new{typeof(factorization),typeof(representor)}(factorization, representor)
    end
end

"""
    TopDown(; factorization=ACA(), representor=nothing)

Build a [`TopDown`](@ref) compressor from keyword arguments.

# Keyword Arguments

  - `factorization = ACA()`: low-rank factorization that selects the pivots
  - `representor = nothing`: [`Representor`](@ref) that reduces the far field
"""
function TopDown(; factorization=AdaptiveCrossApproximation.ACA(), representor=nothing)
    return TopDown(factorization, representor)
end

## testtree

function testbases(
    compressor::TopDown,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = testbuffer(compressor, farmatrix, maxrank)

    tpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    Ftpivots = Vector{Vector{Int}}(undef, numberofnodes(testtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(testtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(testtree(tree)))]
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    nleaves = count(n -> isleaf(testtree(tree), n), 1:numberofnodes(testtree(tree)))
    for level in levels(testtree(tree))
        testclusters = collect(LevelIterator(testtree(tree), level))
        levelstep = max(1, nleaves ÷ length(testclusters))
        @tasks for t in testclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartestbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            Ft = collect(nodes[ptr[t]:(ptr[t + 1] - 1)])
            Ftvalues = H2Trees.values(trialtree(tree), Ft)
            isassigned(Ftpivots, H2Trees.parent(testtree(tree), t)) &&
                append!(Ftvalues, Ftpivots[H2Trees.parent(testtree(tree), t)])
            if !isempty(Ftvalues)
                tvalues = H2Trees.values(testtree(tree), t)

                tpivots[t], Ftpivots[t] = compute_test_pivots!(
                    factorization,
                    representor,
                    farmatrix,
                    tree,
                    tvalues,
                    PreResolvedIndices(Ftvalues),
                    buffer[bufferidx(level)],
                    farbuffer;
                    maxrank=maxrank,
                )
                if isleaf(H2Trees.testtree(tree), t)
                    bases[t] = nestedtestbasis(
                        tvalues, tpivots[t], buffer[bufferidx(level)]
                    )
                end
                next!(pbar; step=levelstep)
            end
        end
        sizehint!(level_transfer_nodes[level], length(testclusters))
        @inbounds for t in testclusters
            if isassigned(tpivots, t)
                !isleaf(H2Trees.testtree(tree), t) && push!(level_transfer_nodes[level], t)
            end
        end
        level != 1 && testtransfermatrices!(
            testtree(tree),
            transfer,
            level_transfer_nodes[level - 1],
            tpivots,
            buffer[bufferidx(level - 1)];
            scheduler=scheduler,
        )
    end
    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(transfer, level_transfer_nodes, testtree(tree))
    return basis_store, transfer_store, tpivots
end

function trialbases(
    compressor::TopDown,
    farmatrix::AbstractKernelMatrix{T},
    tree::BlockTree,
    fardata::FarData;
    scheduler=DynamicScheduler(),
    maxrank=40,
    pbar=Progress(0; enabled=false),
) where {T}
    buffer = trialbuffer(compressor, farmatrix, maxrank)

    Fspivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    spivots = Vector{Vector{Int}}(undef, numberofnodes(trialtree(tree)))
    bases = Vector{Matrix{T}}(undef, numberofnodes(trialtree(tree)))
    transfer = Vector{Vector{Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    level_transfer_nodes = [Int[] for _ in 1:length(levels(trialtree(tree)))]
    ptr = farptr(fardata)
    nodes = farnodes(fardata)
    nleaves = count(n -> isleaf(trialtree(tree), n), 1:numberofnodes(trialtree(tree)))
    for level in levels(trialtree(tree))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        levelstep = max(1, nleaves ÷ length(trialclusters))
        @tasks for s in trialclusters
            @set scheduler = scheduler
            @local begin
                farbuffer = fartrialbuffer(compressor.factorization, farmatrix, maxrank)
                factorization = _build_stateful_factorization(
                    compressor.factorization, farmatrix, maxrank
                )
                representor = _stateful_representor(compressor.representor, maxrank)
            end
            Fs = collect(nodes[ptr[s]:(ptr[s + 1] - 1)])
            Fsvalues = H2Trees.values(testtree(tree), Fs)
            isassigned(Fspivots, H2Trees.parent(trialtree(tree), s)) &&
                append!(Fsvalues, Fspivots[H2Trees.parent(trialtree(tree), s)])
            if !isempty(Fsvalues)
                svalues = H2Trees.values(trialtree(tree), s)
                Fspivots[s], spivots[s] = compute_trial_pivots!(
                    factorization,
                    representor,
                    farmatrix,
                    tree,
                    PreResolvedIndices(Fsvalues),
                    svalues,
                    farbuffer,
                    buffer[bufferidx(level)];
                    maxrank=maxrank,
                )
                if isleaf(H2Trees.trialtree(tree), s)
                    bases[s] = nestedtrialbasis(
                        svalues, spivots[s], buffer[bufferidx(level)]
                    )
                end
                next!(pbar; step=levelstep)
            end
        end
        sizehint!(level_transfer_nodes[level], length(trialclusters))
        @inbounds for s in trialclusters
            if isassigned(spivots, s)
                !isleaf(H2Trees.trialtree(tree), s) && push!(level_transfer_nodes[level], s)
            end
        end
        level != 1 && trialtransfermatrices!(
            trialtree(tree),
            transfer,
            level_transfer_nodes[level - 1],
            spivots,
            buffer[bufferidx(level - 1)];
            scheduler=scheduler,
        )
    end
    basis_store = _build_basis_storage(bases)
    transfer_store = _build_transfer_storage(
        transfer, level_transfer_nodes, trialtree(tree)
    )
    return basis_store, transfer_store, spivots
end
