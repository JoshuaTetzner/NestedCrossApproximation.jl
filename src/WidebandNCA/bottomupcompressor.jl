
function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dirdata::DirectionalData,
    tree::BlockTree,
    buffer::Tuple{K,Channel{K}};
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

    for level in reverse(levels(testtree(tree)))
        @tasks for t in collect(LevelIterator(testtree(tree), level))
            @set ntasks = ntasks
            localbases = Matrix{T}[]
            localtransfermats = Vector{Tuple{Int,Matrix{T}}}[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]
            for dir in directions(dirdata, t)
                Ft = testfarfield(dirdata, testtree(tree), t, dir; islf=islf)
                if Ft != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        tree,
                        t,
                        Ft,
                        buffer[1],
                        buffer[2];
                        maxrank=maxrank,
                    )
                    push!(localpivots, pivs)

                    if isdirectionalroot(dirdata, testtree(tree), t)
                        nestedtestbasis!(localbases, buffer[1], tree, t, pivs)
                    else
                        push!(
                            localtransfermats,
                            testtransfermatrix(
                                t, dir, pivs, buffer[1], pivots, testtree(tree), dirdata
                            ),
                        )
                    end
                end
            end
            localbases != [] &&
                (nestedbases[t] = Dict(directions(dirdata, t) .=> localbases))
            localtransfermats != [] &&
                (transfermatrices[t] = Dict(directions(dirdata, t) .=> localtransfermats))
            if directions(dirdata, t) != []
                pivots[t] = Dict(directions(dirdata, t) .=> localpivots)
            end
        end
    end

    return nestedbases, transfermatrices, pivots
end

function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dirdata::DirectionalData,
    tree::H2Trees.BlockTree,
    buffer::Tuple{Channel{K},K};
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

    for level in reverse(levels(trialtree(tree)))
        @tasks for s in collect(LevelIterator(trialtree(tree), level))
            @set ntasks = ntasks
            localbases = Matrix{T}[]
            localtransfermats = Vector{Tuple{Int,Matrix{T}}}[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]

            for dir in directions(dirdata, s)
                Fs = trialfarfield(dirdata, trialtree(tree), s, dir; islf=islf)
                if Fs != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        tree,
                        Fs,
                        s,
                        buffer[1],
                        buffer[2];
                        maxrank=maxrank,
                    )
                    push!(localpivots, pivs)

                    if isdirectionalroot(dirdata, trialtree(tree), s)
                        nestedtrialbasis!(localbases, buffer[2], tree, s, pivs)
                    else
                        push!(
                            localtransfermats,
                            trialtransfermatrix(
                                s, dir, pivs, buffer[2], pivots, trialtree(tree), dirdata
                            ),
                        )
                    end
                end
            end
            localbases != [] &&
                (nestedbases[s] = Dict(directions(dirdata, s) .=> localbases))
            localtransfermats != [] &&
                (transfermatrices[s] = Dict(directions(dirdata, s) .=> localtransfermats))
            if directions(dirdata, s) != []
                pivots[s] = Dict(directions(dirdata, s) .=> localpivots)
            end
        end
    end

    return nestedbases, transfermatrices, pivots
end
