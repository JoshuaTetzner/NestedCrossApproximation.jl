
function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    Ft::Vector{Vector{Int}},
    eₜ::Vector{Vector{Int}},
    tree::BlockTree,
    dtree::𝒟tree,
    buffer::Tuple{K,Channel{K}};
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
    bottomupfars!(testtree(tree), dtree, Ft, eₜ; ntasks=ntasks)

    for level in reverse(levels(testtree(tree)))
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
            for dir in dirs
                dirFt = Ft[t][findall(x -> x == dir, eₜ[t])]
                if dirFt != []
                    pivs = compress(
                        compressor, farmatrix, tree, t, dirFt, buffer[1], buffer[2]
                    )

                    push!(
                        localblocks,
                        buffer[1][H2Trees.values(testtree(tree), t), 1:length(pivs[1])],
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
        build_butestbases!(
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

function (compressor::BottomUpCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    Fs::Vector{Vector{Int}},
    eₛ::Vector{Vector{Int}},
    tree::H2Trees.BlockTree,
    dtree::𝒟tree,
    buffer::Tuple{Channel{K},K};
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    leveledtransfer = Dict{Int,Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}}[]
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, length(H2Trees.trialtree(tree).nodes)
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, length(H2Trees.trialtree(tree).nodes))
    bottomupfars!(trialtree(tree), dtree, Fs, eₛ; ntasks=ntasks)

    for level in reverse(levels(trialtree(tree)))
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
            for dir in dirs
                dirFs = Fs[s][findall(x -> x == dir, eₛ[s])]
                if dirFs != []
                    pivs = compress(
                        compressor, farmatrix, tree, dirFs, s, buffer[1], buffer[2]
                    )

                    push!(
                        localblocks,
                        buffer[2][1:length(pivs[1]), H2Trees.values(trialtree(tree), s)],
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
        build_butrialbases!(
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
