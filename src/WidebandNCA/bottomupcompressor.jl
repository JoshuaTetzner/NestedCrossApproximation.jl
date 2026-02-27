
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
            rowbuffer = take!(buffer[2])

            localbases = Matrix{T}[]
            localtransfermats = Vector{Tuple{Int,Matrix{T}}}[]
            localpivots = Vector{Tuple{Vector{Int},Vector{Int}}}(
                undef, length(directions(dirdata, t))
            )
            for (diridx, dir) in enumerate(directions(dirdata, t))
                Ft = testfarfield(dirdata, testtree(tree), t, dir; islf=islf)
                Ft == [] && continue
                rowidcs, colidcs = rcindices(compressor, tree, t, Ft)

                # This could be moved into rcindices
                if !isdirectionalroot(dirdata, testtree(tree), t)
                    rowidcs = Int[]
                    for child in ChildIterator(testtree(tree), t)
                        rowidcs = vcat(
                            rowidcs,
                            pivots[child][paternaldirection(dirdata, child, dir)][1],
                        )
                    end
                end
                newmaxrank = min(length(unique(rowidcs)), maxrank)
                newmaxrank = min(
                    newmaxrank, length(H2Trees.values(trialtree(tree), colidcs))
                )
                pivs = testcompress(
                    compressor,
                    farmatrix,
                    unique(rowidcs),
                    colidcs,
                    buffer[1],
                    rowbuffer;
                    maxrank=newmaxrank,
                )
                localpivots[diridx] = pivs

                isdirectionalroot(dirdata, testtree(tree), t) &&
                    (nestedtestbasis!(localbases, buffer[1], tree, t, pivs); continue)
                push!(
                    localtransfermats,
                    testtransfermatrix(
                        t, dir, pivs, buffer[1], pivots, testtree(tree), dirdata
                    ),
                )
            end
            localbases != [] &&
                (nestedbases[t] = Dict(directions(dirdata, t) .=> localbases);
                pivots[t] = Dict(directions(dirdata, t) .=> localpivots))
            localtransfermats != [] &&
                (transfermatrices[t] = Dict(directions(dirdata, t) .=> localtransfermats);
                pivots[t] = Dict(directions(dirdata, t) .=> localpivots))

            put!(buffer[2], rowbuffer)
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
    transfermatrices = Vector{Dict{Int,Vector{Tuple{Int,Matrix{T}}}}}(
        undef, numberofnodes(trialtree(tree))
    )
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, numberofnodes(trialtree(tree))
    )

    for level in reverse(levels(trialtree(tree)))
        @tasks for s in collect(LevelIterator(trialtree(tree), level))
            @set ntasks = ntasks
            colbuffer = take!(buffer[1])
            localbases = Matrix{T}[]
            localtransfermats = Vector{Tuple{Int,Matrix{T}}}[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]
            for dir in directions(dirdata, s)
                Fs = trialfarfield(dirdata, trialtree(tree), s, dir; islf=islf)
                Fs == [] && continue
                rowidcs, colidcs = rcindices(compressor, tree, Fs, s)
                if !isdirectionalroot(dirdata, trialtree(tree), s)
                    colidcs = Int[]
                    for child in ChildIterator(trialtree(tree), s)
                        colidcs = vcat(
                            colidcs,
                            pivots[child][paternaldirection(dirdata, child, dir)][2],
                        )
                    end
                end
                newmaxrank = min(length(unique(colidcs)), maxrank)
                newmaxrank = min(
                    newmaxrank, length(H2Trees.values(testtree(tree), rowidcs))
                )
                pivs = trialcompress(
                    compressor,
                    farmatrix,
                    rowidcs,
                    unique(colidcs),
                    colbuffer,
                    buffer[2];
                    maxrank=newmaxrank,
                )
                push!(localpivots, pivs)

                isdirectionalroot(dirdata, trialtree(tree), s) &&
                    (nestedtrialbasis!(localbases, buffer[2], tree, s, pivs); continue)
                push!(
                    localtransfermats,
                    trialtransfermatrix(
                        s, dir, pivs, buffer[2], pivots, trialtree(tree), dirdata
                    ),
                )
            end
            localbases != [] &&
                (nestedbases[s] = Dict(directions(dirdata, s) .=> localbases);
                pivots[s] = Dict(directions(dirdata, s) .=> localpivots))
            localtransfermats != [] &&
                (transfermatrices[s] = Dict(directions(dirdata, s) .=> localtransfermats);
                pivots[s] = Dict(directions(dirdata, s) .=> localpivots))
            put!(buffer[1], colbuffer)
        end
    end

    return nestedbases, transfermatrices, pivots
end
