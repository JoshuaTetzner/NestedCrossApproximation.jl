
function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dirdata::DirectionalData,
    tree::BlockTree,
    buffer::Tuple{Tuple{K,K},Channel{K}};
    islf=islf(wavenumber(farmatrix.operator)),
    ntasks=Threads.nthreads(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    nestedbases = Vector{Dict{Int,Matrix{T}}}(undef, numberofnodes(testtree(tree)))
    transfermatrices = Vector{Dict{Int,Vector{Tuple{Int,K}}}}(
        undef, numberofnodes(testtree(tree))
    )
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, numberofnodes(testtree(tree))
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, numberofnodes(testtree(tree)))

    for level in levels(testtree(tree))
        @tasks for t in collect(LevelIterator(testtree(tree), level))
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localbases = Matrix{T}[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]
            for dir in directions(dirdata, t)
                Ftvals = H2Trees.values(
                    trialtree(tree), dirdata.F[t][findall(x -> x == dir, dirdata.𝓔[t])]
                )

                append!(
                    Ftvals,
                    inheritedtrialpivots(
                        dirdata, pivots, testtree(tree), t, dir; islf=islf
                    ),
                )

                if Ftvals != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        H2Trees.values(testtree(tree), t),
                        Ftvals,
                        buffer[1][bufferidx(level)],
                        buffer[2];
                        maxrank=maxrank,
                    )
                    push!(localpivots, pivs)

                    if isdirectionalroot(dirdata, testtree(tree), t)
                        nestedtestbasis!(
                            localbases, buffer[1][bufferidx(level)], tree, t, pivs
                        )
                    else
                        push!(
                            localblocks,
                            buffer[1][bufferidx(level)][
                                H2Trees.values(testtree(tree), t), 1:length(pivs[2])
                            ],
                        )
                    end
                end
            end
            localbases != [] &&
                (nestedbases[t] = Dict(directions(dirdata, t) .=> localbases))
            if directions(dirdata, t) != []
                pivots[t] = Dict(directions(dirdata, t) .=> localpivots)
                localblocks != [] &&
                    (blocks[t] = Dict(directions(dirdata, t) .=> localblocks))
            end
        end
        level != 1 && testtransfermatrices!(
            transfermatrices,
            level - 1,
            blocks,
            pivots,
            testtree(tree),
            dirdata;
            ntasks=ntasks,
        )
    end

    return nestedbases, transfermatrices, pivots
end

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dirdata::DirectionalData,
    tree::H2Trees.BlockTree,
    buffer::Tuple{Channel{K},Tuple{K,K}};
    islf=islf(wavenumber(farmatrix.operator)),
    ntasks=Threads.nthreads(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    nestedbases = Vector{Dict{Int,Matrix{T}}}(undef, numberofnodes(trialtree(tree)))
    transfermatrices = Vector{Dict{Int,Vector{Tuple{Int,K}}}}(
        undef, numberofnodes(trialtree(tree))
    )
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, numberofnodes(trialtree(tree))
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, numberofnodes(trialtree(tree)))

    for level in levels(trialtree(tree))
        @tasks for s in collect(LevelIterator(trialtree(tree), level))
            @set ntasks = ntasks

            localblocks = Matrix{T}[]
            localbases = Matrix{T}[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]

            for dir in directions(dirdata, s)
                Fsvals = H2Trees.values(
                    testtree(tree), dirdata.F[s][findall(x -> x == dir, dirdata.𝓔[s])]
                )

                append!(
                    Fsvals,
                    inheritedtestpivots(
                        dirdata, pivots, trialtree(tree), s, dir; islf=islf
                    ),
                )

                if Fsvals != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        Fsvals,
                        H2Trees.values(trialtree(tree), s),
                        buffer[1],
                        buffer[2][bufferidx(level)];
                        maxrank=maxrank,
                    )

                    push!(localpivots, pivs)

                    if isdirectionalroot(dirdata, trialtree(tree), s)
                        nestedtrialbasis!(
                            localbases, buffer[2][bufferidx(level)], tree, s, pivs
                        )
                    else
                        push!(
                            localblocks,
                            buffer[2][bufferidx(level)][
                                1:length(pivs[1]), H2Trees.values(trialtree(tree), s)
                            ],
                        )
                    end
                end
            end
            localbases != [] &&
                (nestedbases[s] = Dict(directions(dirdata, s) .=> localbases))
            if directions(dirdata, s) != []
                pivots[s] = Dict(directions(dirdata, s) .=> localpivots)
                localblocks != [] &&
                    (blocks[s] = Dict(directions(dirdata, s) .=> localblocks))
            end
        end
        level != 1 && trialtransfermatrices!(
            transfermatrices,
            level - 1,
            blocks,
            pivots,
            trialtree(tree),
            dirdata;
            ntasks=ntasks,
        )
    end

    return nestedbases, transfermatrices, pivots
end
