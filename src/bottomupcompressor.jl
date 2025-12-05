
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
    nFtvalues = length(H2Trees.values(tree.testcluster, Ft))
    npivots, rowpivots, colpivots = compressor.lrf(
        farmatrix,
        view(colbuffer, tvalues, 1:maxrank),
        rowbuffer,
        min(maxrank, min(length(tvalues), nFtvalues));
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
    tree::H2Trees.BlockTree,
    buffer::Tuple{K,Channel{K}};
    ntasks=1,
    isnear=H2Trees.isnear,
    maxrank=40,
) where {T,K<:Matrix{T}}
    nestedbases = Vector{K}(undef, numberofnodes(testtree(tree)))
    transfermatrices = Vector{Vector{K}}(undef, numberofnodes(testtree(tree)))
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, length(testtree(tree).nodes))

    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)

    for level in reverse(levels(testtree(tree)))
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = ntasks

            Ft = collect(iterator(trialtree(tree), testtree(tree), t))
            !(parent(testtree(tree), t) == 0) && (
                Ft = Vector{Int}(
                    vcat(
                        Ft,
                        mapreduce(vcat, ParentUpwardsIterator(testtree(tree), t)) do tp
                            collect(iterator(trialtree(tree), testtree(tree), tp))
                        end,
                    ),
                )
            )

            if Ft != []
                pivots[t] = compress(
                    compressor, farmatrix, tree, t, Ft, buffer[1], buffer[2]
                )

                if H2Trees.firstchild(testtree(tree), t) == 0
                    nestedbases[t] =
                        buffer[1][
                            H2Trees.values(testtree(tree), t), 1:length(pivots[t][1])
                        ] / buffer[1][pivots[t][1], 1:length(pivots[t][1])]
                else
                    transfermatrices[t] = map(ChildIterator(testtree(tree), t)) do tc
                        buffer[1][pivots[tc][1], 1:length(pivots[t][2])] /
                        buffer[1][pivots[t][1], 1:length(pivots[t][2])]
                    end
                end
            end
        end
    end

    return nestedbases, transfermatrices, pivots
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
    nFsvalues = length(H2Trees.values(tree.testcluster, Fs))
    npivots, rowpivots, colpivots = compressor.lrf(
        farmatrix,
        colbuffer,
        view(rowbuffer, 1:maxrank, svalues),
        min(maxrank, min(length(svalues), nFsvalues));
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
    tree::H2Trees.BlockTree,
    buffer::Tuple{Channel{K},K};
    ntasks=1,
    isnear=H2Trees.isnear,
    maxrank=40,
) where {T,K<:Matrix{T}}
    nestedbases = Vector{K}(undef, numberofnodes(trialtree(tree)))
    transfermatrices = Vector{Vector{K}}(undef, numberofnodes(testtree(tree)))
    pivots = Vector{Tuple{Vector{Int},Vector{Int}}}(undef, length(testtree(tree).nodes))

    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)

    for level in reverse(levels(trialtree(tree)))
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set ntasks = ntasks

            Fs = collect(iterator(testtree(tree), trialtree(tree), s))
            !(parent(trialtree(tree), s) == 0) && (
                Fs = Vector{Int}(
                    vcat(
                        Fs,
                        mapreduce(vcat, ParentUpwardsIterator(trialtree(tree), s)) do sp
                            collect(iterator(testtree(tree), trialtree(tree), sp))
                        end,
                    ),
                )
            )
            if Fs != []
                pivots[s] = compress(
                    compressor, farmatrix, tree, Fs, s, buffer[1], buffer[2]
                )

                if H2Trees.isleaf(trialtree(tree), s)
                    nestedbases[s] =
                        buffer[2][1:length(pivots[s][1]), pivots[s][2]] \ buffer[2][
                            1:length(pivots[s][1]), H2Trees.values(trialtree(tree), s)
                        ]
                else
                    transfermatrices[s] = map(ChildIterator(trialtree(tree), s)) do sc
                        buffer[2][1:length(pivots[s][2]), pivots[s][2]] \
                        buffer[2][1:length(pivots[s][2]), pivots[sc][2]]
                    end
                end
            end
        end
    end

    return nestedbases, transfermatrices, pivots
end
