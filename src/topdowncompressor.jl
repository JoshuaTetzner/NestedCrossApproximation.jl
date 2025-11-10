## testtree

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    farinteractions::Vector{Vector{Int}},
    tree::BlockTree,
    buffer::Tuple{Tuple{K,K},Channel{K}};
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,H2BasisBlock{Int,T}}[]
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, numberofnodes(testtree(tree)))
    for level in levels(testtree(tree))
        transferidcs = Int[]
        transfer = H2BasisBlock{Int,T}[]
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for testcluster in testclusters
            @set ntasks = ntasks
            Ftvalues = H2Trees.values(trialtree(tree), farinteractions[testcluster])
            isassigned(pivots, H2Trees.parent(testtree(tree), testcluster)) &&
                append!(Ftvalues, pivots[H2Trees.parent(testtree(tree), testcluster)][2])
            Ftvalues != [] && (
                pivots[testcluster] = compress(
                    compressor,
                    farmatrix,
                    H2Trees.values(testtree(tree), testcluster),
                    Ftvalues,
                    buffer[1][bufferidx(level)],
                    buffer[2],
                )
            )
        end
        build_testbases!(
            transfer,
            transferidcs,
            bases,
            basesidcs,
            buffer[1],
            testclusters,
            pivots,
            level,
            testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end

## trialtree

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    farinteractions::Vector{Vector{Int}},
    tree::BlockTree,
    buffer::Tuple{Channel{K},Tuple{K,K}};
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,H2BasisBlock{Int,T}}[]
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, numberofnodes(trialtree(tree)))

    for level in levels(trialtree(tree))
        transferidcs = Int[]
        transfer = H2BasisBlock{Int,T}[]
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for trialcluster in trialclusters
            @set ntasks = ntasks
            Fsvalues = H2Trees.values(testtree(tree), farinteractions[trialcluster])
            isassigned(pivots, H2Trees.parent(trialtree(tree), trialcluster)) &&
                append!(Fsvalues, pivots[H2Trees.parent(trialtree(tree), trialcluster)][1])
            Fsvalues != [] && (
                pivots[trialcluster] = compress(
                    compressor,
                    farmatrix,
                    Fsvalues,
                    H2Trees.values(trialtree(tree), trialcluster),
                    buffer[1],
                    buffer[2][bufferidx(level)],
                )
            )
        end
        build_trialbases!(
            transfer,
            transferidcs,
            bases,
            basesidcs,
            buffer[2],
            trialclusters,
            pivots,
            level,
            trialtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end
