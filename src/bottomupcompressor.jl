
#testtree

function rcindices(
    compressor::BottomUpCompressor{iACA{RP,CP,C},R},
    tree::H2Trees.BlockTree,
    t::Int,
    Ft::Vector{Int},
) where {RP,CP<:MimicryPivoting,C,R}
    if isnothing(compressor.representor)
        Ftvalues = Int[]
        for s in Ft
            append!(Ftvalues, H2Trees.values(H2Trees.trialtree(tree), s))
        end
        return H2Trees.values(H2Trees.testtree(tree), t), Ftvalues
    else
        return error("Not implemented")
    end
end

function rcindices(
    compressor::BottomUpCompressor{iACA{RP,CP,C},R},
    tree::H2Trees.BlockTree,
    t::Int,
    Ft::Vector{Int},
) where {RP,CP<:TreeMimicryPivoting,C,R}
    isnothing(compressor.representor) &&
        (return H2Trees.values(H2Trees.testtree(tree), t), Ft)
    return error("Not implemented")
end

function compress(
    compressor::BottomUpCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    tree,
    t::Int,
    Ft::Vector{Int},
    colbuffer::K,
    rowchannel::Channel{K};
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:iACA,RP}
    tvalues, Ftvalues = rcindices(compressor, tree, t, Ft)
    rowbuffer = take!(rowchannel)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    colbuffer[tvalues, 1:maxrank] .= 0.0

    npivots, rowpivots, colpivots = compressor.lrf(
        farmatrix,
        view(colbuffer, tvalues, 1:maxrank),
        rowbuffer,
        min(maxrank, length(tvalues));
        rows=rows,
        cols=cols,
        rowidcs=tvalues,
        colidcs=Ftvalues,
    )
    colbuffer[tvalues, 1:npivots] =
        colbuffer[tvalues, 1:npivots] * rowbuffer[1:npivots, 1:npivots]
    rowbuffer[1:npivots, 1:npivots] .= 0.0

    put!(rowchannel, rowbuffer)

    return rowpivots, colpivots
end

function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    farinteractions::Vector{Vector{Int}},
    tree::H2Trees.BlockTree,
    buffer::Tuple{K,Channel{K}};
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = NestedCrossApproximation.H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(
        undef, length(H2Trees.testtree(tree).nodes)
    )
    for level in reverse(H2Trees.levels(H2Trees.testtree(tree)))
        transferidcs = Int[]
        transfer = NestedCrossApproximation.H2BasisBlock{Int,T}[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = ntasks
            nodes = vcat(
                t, collect(H2Trees.ParentUpwardsIterator(H2Trees.testtree(tree), t))
            )
            nodes != [] ? (fars = reduce(vcat, farinteractions[nodes])) : (fars = [])
            fars != [] && (
                pivots[t] = compress(
                    compressor, farmatrix, tree, t, fars, buffer[1], buffer[2]
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
            H2Trees.testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), reverse!(leveledtransfer), pivots
end

# trialtree

function rcindices(
    compressor::BottomUpCompressor{iACA{RP,CP,C},R},
    tree::H2Trees.BlockTree,
    Fs::Vector{Int},
    s::Int,
) where {RP<:TreeMimicryPivoting,CP,C,R}
    isnothing(compressor.representor) &&
        return (Fs, H2Trees.values(H2Trees.trialtree(tree), s))
    return error("Not implemented")
end

function rcindices(
    compressor::BottomUpCompressor{iACA{RP,CP,C},R},
    tree::H2Trees.BlockTree,
    Fs::Vector{Int},
    s::Int,
) where {RP<:MimicryPivoting,CP,C,R}
    if isnothing(compressor.representor)
        Fsvalues = Int[]
        for t in Fs
            append!(Fsvalues, H2Trees.values(H2Trees.testtree(tree), t))
        end
        return (Fsvalues, H2Trees.values(H2Trees.trialtree(tree), s))
    else
        return error("Not implemented")
    end
end

function compress(
    compressor::BottomUpCompressor{LRF,RP},
    farmatrix::AbstractKernelMatrix{T},
    tree,
    Fs::Vector{Int},
    s::Int,
    colchannel::Channel{K},
    rowbuffer::K;
    maxrank=40,
) where {T,K<:Matrix{T},LRF<:iACA,RP}
    Fsvalues, svalues = rcindices(compressor, tree, Fs, s)
    colbuffer = take!(colchannel)
    rows = zeros(Int, maxrank)
    cols = zeros(Int, maxrank)

    rowbuffer[1:maxrank, svalues] .= 0.0

    npivots, rowpivots, colpivots = compressor.lrf(
        farmatrix,
        colbuffer,
        view(rowbuffer, 1:maxrank, svalues),
        min(maxrank, length(svalues));
        rows=rows,
        cols=cols,
        rowidcs=Fsvalues,
        colidcs=svalues,
    )
    rowbuffer[1:npivots, svalues] =
        colbuffer[1:npivots, 1:npivots] * rowbuffer[1:npivots, svalues]
    colbuffer[1:npivots, 1:npivots] .= 0.0

    put!(colchannel, colbuffer)

    return rowpivots, colpivots
end

function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    farinteractions::Vector{Vector{Int}},
    tree::H2Trees.BlockTree,
    buffer::Tuple{Channel{K},K};
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = NestedCrossApproximation.H2BasisBlock{Int,T}[]
    leveledtransfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(
        undef, length(H2Trees.testtree(tree).nodes)
    )
    for level in reverse(H2Trees.levels(H2Trees.trialtree(tree)))
        transferidcs = Int[]
        transfer = NestedCrossApproximation.H2BasisBlock{Int,T}[]
        trialclusters = collect(H2Trees.LevelIterator(H2Trees.trialtree(tree), level))
        @tasks for s in trialclusters
            @set ntasks = ntasks
            nodes = vcat(
                s, collect(H2Trees.ParentUpwardsIterator(H2Trees.trialtree(tree), s))
            )
            nodes != [] ? (fars = reduce(vcat, farinteractions[nodes])) : (fars = [])
            fars != [] && (
                pivots[s] = compress(
                    compressor, farmatrix, tree, fars, s, buffer[1], buffer[2]
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
            H2Trees.testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end

    return Dict(basesidcs .=> bases), reverse!(leveledtransfer), pivots
end
