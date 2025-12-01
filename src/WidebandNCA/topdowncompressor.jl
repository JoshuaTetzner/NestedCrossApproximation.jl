
function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    Ft::Vector{Vector{Int}},
    eₜ::Vector{Vector{Int}},
    tree::BlockTree,
    dtree::𝒟tree,
    buffer::Tuple{Tuple{K,K},Channel{K}};
    ntasks=Threads.nthreads(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    leveledtransfer = Dict{Int,Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}}[]
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, length(H2Trees.testtree(tree).nodes)
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, length(H2Trees.testtree(tree).nodes))

    for level in levels(testtree(tree))
        transferidcs = Int[]
        transfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
        testclusters = collect(LevelIterator(testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localdirs = Int[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]

            # add paternal directions
            dirs = unique(eₜ[t])
            (!(dirs == [0]) && isassigned(pivots, parent(testtree(tree), t))) &&
                for eₜₜ in keys(pivots[parent(testtree(tree), t)])#eₜ[parent(testtree(tree), t)]
                    !in(parent(dtree, eₜₜ), dirs) && push!(dirs, parent(dtree, eₜₜ))
                end

            for dir in dirs
                Ftvals = H2Trees.values(
                    trialtree(tree), Ft[t][findall(x -> x == dir, eₜ[t])]
                )
                if isassigned(pivots, parent(testtree(tree), t)) && dir != 0
                    for child in children(dtree, dir)
                        haskey(pivots[parent(testtree(tree), t)], child) &&
                            append!(Ftvals, pivots[parent(testtree(tree), t)][child][2])
                    end
                end
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

                    push!(
                        localblocks,
                        buffer[1][bufferidx(level)][
                            H2Trees.values(testtree(tree), t), 1:length(pivs[1])
                        ],
                    )

                    push!(localpivots, pivs)
                    push!(localdirs, dir)
                end
            end

            if localdirs != []
                pivots[t] = Dict(localdirs .=> localpivots)
                blocks[t] = Dict(localdirs .=> localblocks)
            end
        end
        build_testbases!(
            transfer,
            transferidcs,
            bases,
            basesidcs,
            blocks,
            pivots,
            level,
            dtree,
            testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end
    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Vector{Int}},
    eₛ::Vector{Vector{Int}},
    tree::H2Trees.BlockTree,
    dtree::𝒟tree,
    buffer::Tuple{Channel{K},Tuple{K,K}};
    islf=islf(wavenumber(farmatrix.operator)),
    ntasks=Threads.nthreads(),
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    leveledtransfer = Dict{Int,Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}}[]
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, length(H2Trees.trialtree(tree).nodes)
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, length(H2Trees.trialtree(tree).nodes))
    for level in levels(trialtree(tree))
        transferidcs = Int[]
        transfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localdirs = Int[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]

            # add paternal directions
            dirs = unique(eₛ[s])
            (!(dirs == [0]) && isassigned(pivots, parent(trialtree(tree), s))) &&
                for eₛₛ in keys(pivots[parent(trialtree(tree), s)])
                    !in(parent(dtree, eₛₛ), dirs) && push!(dirs, parent(dtree, eₛₛ))
                end

            for dir in dirs
                Fsvals = H2Trees.values(
                    testtree(tree), Fs[s][findall(x -> x == dir, eₛ[s])]
                )
                if isassigned(pivots, parent(trialtree(tree), s)) && dir != 0
                    for child in children(dtree, dir)
                        haskey(pivots[H2Trees.parent(tree.trialcluster, s)], child) &&
                            append!(Fsvals, pivots[parent(trialtree(tree), s)][child][1])
                    end
                end
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

                    push!(
                        localblocks,
                        buffer[2][bufferidx(level)][
                            1:length(pivs[1]), H2Trees.values(trialtree(tree), s)
                        ],
                    )
                    push!(localpivots, pivs)
                    push!(localdirs, dir)
                end
            end

            if localdirs != []
                pivots[s] = Dict(localdirs .=> localpivots)
                blocks[s] = Dict(localdirs .=> localblocks)
            end
        end
        level != Dict() && build_trialbases!(
            transfer,
            transferidcs,
            bases,
            basesidcs,
            blocks,
            pivots,
            level,
            dtree,
            trialtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end
    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end
