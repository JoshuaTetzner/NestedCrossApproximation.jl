#=struct TopDownCompressor{LowRankFactorizationType,RepresentorType}
    lrf::LowRankFactorizationType
    representor::RepresentorType

    function TopDownCompressor(lrf, representor)
        return new{typeof(lrf),typeof(representor)}(lrf, representor)
    end
end

function TopDownCompressor(;
    factorization=AdaptiveCrossApproximation.ACA(), representor=nothing
)
    return TopDownCompressor(factorization, representor)
end=#

## testtree

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    tree::H2Trees.BlockTree,
    buffer::Tuple{Tuple{K,K},Channel{K}};
    isnear=H2Trees.isnear,
    ntasks=1,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = NestedCrossApproximation.H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(
        undef, length(H2Trees.testtree(tree).nodes)
    )
    for level in H2Trees.levels(H2Trees.testtree(tree))
        transferidcs = Int[]
        transfer = NestedCrossApproximation.H2BasisBlock{Int,T}[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for testcluster in testclusters
            @set ntasks = ntasks
            Ft = Int[]
            for s in iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), testcluster)
                append!(Ft, H2Trees.values(H2Trees.trialtree(tree), s))
            end
            isassigned(pivots, H2Trees.parent(H2Trees.testtree(tree), testcluster)) &&
                append!(Ft, pivots[H2Trees.parent(H2Trees.testtree(tree), testcluster)][2])
            Ft != [] && (
                pivots[testcluster] = compress(
                    compressor,
                    farmatrix,
                    H2Trees.values(H2Trees.testtree(tree), testcluster),
                    Ft,
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
            H2Trees.testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end
#=
function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    t::Vector{Int},
    Ft::Vector{Int},
    colbuffer::K,
    rowchannel::Channel{K},
) where {T,K<:Matrix{T}}
    !isnothing(compressor.representor) && (t, Ft=compressor.representor(t, Ft))
    rowbuffer = take!(rowchannel)
    rows = zeros(Int, 40)
    cols = zeros(Int, 40)

    #comp = compressor.lrf(t, Ft)
    colbuffer[t, 1:40] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, t, 1:40),
        view(rowbuffer, 1:40, Ft),
        min(40, min(length(t), length(Ft)));
        rows=rows,
        cols=cols,
        rowidcs=t,
        colidcs=Ft,
    )
    colbuffer[t, 1:npivots] =
        colbuffer[t, 1:npivots] * rowbuffer[1:npivots, cols[1:npivots]]

    rowbuffer[1:npivots, Ft] .= 0.0
    put!(rowchannel, rowbuffer)

    return rows[1:npivots], cols[1:npivots]
end
=#
## trialtree

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    tree::H2Trees.BlockTree,
    buffer::Tuple{Channel{K},Tuple{K,K}};
    ntasks=1,
    isnear=H2Trees.inear,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = NestedCrossApproximation.H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(
        undef, length(H2Trees.trialtree(tree).nodes)
    )

    for level in H2Trees.levels(H2Trees.trialtree(tree))
        transferidcs = Int[]
        transfer = NestedCrossApproximation.H2BasisBlock{Int,T}[]
        trialclusters = collect(H2Trees.LevelIterator(H2Trees.trialtree(tree), level))
        @tasks for trialcluster in trialclusters
            @set ntasks = ntasks
            Fs = Int[]
            for t in iterator(H2Trees.trialtree(tree), H2Trees.testtree(tree), trialcluster)
                append!(Fs, H2Trees.values(H2Trees.testtree(tree), t))
            end
            isassigned(pivots, H2Trees.parent(H2Trees.trialtree(tree), trialcluster)) &&
                append!(
                    Fs, pivots[H2Trees.parent(H2Trees.trialtree(tree), trialcluster)][1]
                )
            Fs != [] && (
                pivots[trialcluster] = compress(
                    compressor,
                    farmatrix,
                    Fs,
                    H2Trees.values(H2Trees.trialtree(tree), trialcluster),
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
            H2Trees.trialtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end
#=
function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Int},
    s::Vector{Int},
    colchannel::Channel{K},
    rowbuffer::K,
) where {T,K<:Matrix{T}}
    !isnothing(compressor.representor) && (s, Fs=compressor.representor(s, Fs))
    colbuffer = take!(colchannel)
    rows = zeros(Int, 40)
    cols = zeros(Int, 40)

    #comp = compressor.lrf(Fs, s)
    rowbuffer[1:40, s] .= 0.0
    npivots = compressor.lrf(
        farmatrix,
        view(colbuffer, Fs, 1:40),
        view(rowbuffer, 1:40, s),
        min(40, min(length(s), length(Fs)));
        rows=rows,
        cols=cols,
        rowidcs=Fs,
        colidcs=s,
    )
    rowbuffer[1:npivots, s] =
        colbuffer[rows[1:npivots], 1:npivots] * rowbuffer[1:npivots, s]
    colbuffer[Fs, 1:npivots] .= 0.0
    put!(colchannel, colbuffer)

    return rows[1:npivots], cols[1:npivots]
end
=#
