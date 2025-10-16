
function testfars(
    dtree::𝒟tree,
    tree::H2Trees.BlockTree,
    fars::Vector{Vector{Tuple{Int,Int}}},
    dirs::Vector{Vector{Int}},
    islf,
)
    testdirfars = Vector{Dict{Int,Dict{Int,Vector{Int}}}}(undef, length(fars))
    for level in H2Trees.levels(H2Trees.testtree(tree))
        levelcluster = Int[]
        leveltestdirfars = Dict{Int,Vector{Int}}[]
        for t in H2Trees.LevelIterator(H2Trees.testtree(tree), level)
            islf(tree, level) && break
            faridcs = findall(x -> x[1] == t, fars[level])

            inherdirs = Int[]
            if level > 1 &&
                haskey(testdirfars[level - 1], H2Trees.parent(tree.testcluster, t))
                for dir in keys(testdirfars[level - 1][H2Trees.parent(tree.testcluster, t)])
                    push!(inherdirs, parent(dtree, dir))
                end
            end
            tdirs = [dir for dir in dirs[level][faridcs]]
            localdirs = unique(vcat(tdirs, inherdirs))
            localfars = [Int[] for _ in 1:length(localdirs)]
            for faridx in faridcs
                push!(
                    localfars[findfirst(x -> x == dirs[level][faridx], localdirs)],
                    fars[level][faridx][2],
                )
            end
            if localdirs != []
                push!(levelcluster, t)
                push!(leveltestdirfars, Dict(localdirs .=> localfars))
            end
        end
        testdirfars[level] = Dict(levelcluster .=> leveltestdirfars)
    end
    return testdirfars
end

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dtree,
    tree::H2Trees.BlockTree,
    fars::Vector{Vector{Tuple{Int,Int}}},
    dirs::Vector{Vector{Int}},
    buffer::Tuple{Tuple{K,K},Channel{K}};
    islf=islf(wavenumber(farmatrix.operator)),
    ntasks=1,
    maxrank=40,
) where {T,K<:Matrix{T}}
    basesidcs = Int[]
    bases = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
    leveledtransfer = Dict{Int,Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}}[]
    pivots = Vector{Dict{Int,Tuple{Vector{Int},Vector{Int}}}}(
        undef, length(H2Trees.testtree(tree).nodes)
    )
    blocks = Vector{Dict{Int,Matrix{T}}}(undef, length(H2Trees.testtree(tree).nodes))
    dirfars = testfars(dtree, tree, fars, dirs, islf)
    nl = 0
    for d in dirfars
        d != Dict() && (nl += 1)
    end

    #compressor.lrf.convergence.estimator.tol =
    #    compressor.lrf.convergence.estimator.tol / max(1, nl)
    #println("NewTol: ", compressor.lrf.convergence.estimator.tol)

    for (levelidx, level) in enumerate(dirfars)
        transferidcs = Int[]
        transfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
        #testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in collect(keys(level))
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localdirs = Int[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]
            for (dir, Ft) in level[t]
                Ftvalues = Int[]
                for s in Ft
                    append!(Ftvalues, H2Trees.values(H2Trees.trialtree(tree), s))
                end
                if isassigned(pivots, H2Trees.parent(tree.testcluster, t))
                    for child in children(dtree, dir)
                        if haskey(pivots[H2Trees.parent(tree.testcluster, t)], child)
                            append!(
                                Ftvalues,
                                pivots[H2Trees.parent(H2Trees.testtree(tree), t)][child][2],
                            )
                        end
                    end
                end
                if Ftvalues != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        H2Trees.values(H2Trees.testtree(tree), t),
                        Ftvalues,
                        buffer[1][bufferidx(levelidx)],
                        buffer[2];
                        maxrank=maxrank,
                    )

                    push!(
                        localblocks,
                        buffer[1][bufferidx(levelidx)][
                            H2Trees.values(H2Trees.testtree(tree), t), 1:length(pivs[1])
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
        level != Dict() && build_testbases!(
            transfer,
            transferidcs,
            bases,
            basesidcs,
            blocks,
            collect(keys(level)),
            pivots,
            levelidx,
            dtree,
            H2Trees.testtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end
    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end

function trialfars(
    dtree::𝒟tree,
    tree::H2Trees.BlockTree,
    fars::Vector{Vector{Tuple{Int,Int}}},
    dirs::Vector{Vector{Int}},
    islf,
)
    trialdirfars = Vector{Dict{Int,Dict{Int,Vector{Int}}}}(undef, length(fars))
    for level in H2Trees.levels(H2Trees.trialtree(tree))
        levelcluster = Int[]
        leveltrialdirfars = Dict{Int,Vector{Int}}[]
        for s in H2Trees.LevelIterator(H2Trees.trialtree(tree), level)
            islf(tree, level) && break
            faridcs = findall(x -> x[2] == s, fars[level])

            inherdirs = Int[]
            if level > 1 &&
                haskey(trialdirfars[level - 1], H2Trees.parent(tree.trialcluster, s))
                for dir in
                    keys(trialdirfars[level - 1][H2Trees.parent(tree.trialcluster, s)])
                    push!(inherdirs, parent(dtree, dir))
                end
            end
            sdirs = [dir for dir in dirs[level][faridcs]]
            localdirs = unique(vcat(sdirs, inherdirs))
            localfars = [Int[] for _ in 1:length(localdirs)]
            #faridcs == [] && continue
            for faridx in faridcs
                push!(
                    localfars[findfirst(x -> x == dirs[level][faridx], localdirs)],
                    fars[level][faridx][1],
                )
            end
            if localdirs != []
                push!(levelcluster, s)
                push!(leveltrialdirfars, Dict(localdirs .=> localfars))
            end
        end
        trialdirfars[level] = Dict(levelcluster .=> leveltrialdirfars)
    end
    return trialdirfars
end

function (compressor::TopDownCompressor)(
    farmatrix::AbstractKernelMatrix{T},
    dtree::𝒟tree,
    tree::H2Trees.BlockTree,
    fars::Vector{Vector{Tuple{Int,Int}}},
    dirs::Vector{Vector{Int}},
    buffer::Tuple{Channel{K},Tuple{K,K}};
    islf=islf(wavenumber(farmatrix.operator)),
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
    dirfars = trialfars(dtree, tree, fars, dirs, islf)

    nl = 0
    for d in dirfars
        d != Dict() && (nl += 1)
    end

    #compressor.lrf.convergence.estimator.tol =
    #    compressor.lrf.convergence.estimator.tol / max(1, nl)
    #println("NewTol: ", compressor.lrf.convergence.estimator.tol)

    for (levelidx, level) in enumerate(dirfars)
        transferidcs = Int[]
        transfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
        @tasks for s in collect(keys(level))
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localdirs = Int[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]
            for (dir, Fs) in level[s]
                Fsvalues = Int[]
                for t in Fs
                    append!(Fsvalues, H2Trees.values(H2Trees.testtree(tree), t))
                end
                if isassigned(pivots, H2Trees.parent(tree.trialcluster, s))
                    for child in children(dtree, dir)
                        haskey(pivots[H2Trees.parent(tree.trialcluster, s)], child) &&
                            append!(
                                Fsvalues,
                                pivots[H2Trees.parent(H2Trees.trialtree(tree), s)][child][1],
                            )
                    end
                end
                if Fsvalues != []
                    pivs = compress(
                        compressor,
                        farmatrix,
                        Fsvalues,
                        H2Trees.values(H2Trees.trialtree(tree), s),
                        buffer[1],
                        buffer[2][bufferidx(levelidx)];
                        maxrank=maxrank,
                    )

                    push!(
                        localblocks,
                        buffer[2][bufferidx(levelidx)][
                            1:length(pivs[1]), H2Trees.values(H2Trees.trialtree(tree), s)
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
            collect(keys(level)),
            pivots,
            levelidx,
            dtree,
            H2Trees.trialtree(tree);
            ntasks=ntasks,
        )
        push!(leveledtransfer, Dict(transferidcs .=> transfer))
    end
    return Dict(basesidcs .=> bases), leveledtransfer, pivots
end
