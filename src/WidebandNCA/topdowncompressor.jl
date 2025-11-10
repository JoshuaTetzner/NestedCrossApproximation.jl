
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
    Ft::Vector{Vector{Int}},
    eₜ::Vector{Vector{Int}},
    buffer::Tuple{Tuple{K,K},Channel{K}};
    islf=islf(wavenumber(farmatrix.operator)),
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
    #dirfars = testfars(dtree, tree, Ft, eₜ, islf)
    #nl = 0
    #for d in dirfars
    #    d != Dict() && (nl += 1)
    #end

    #compressor.lrf.convergence.estimator.tol =
    #    compressor.lrf.convergence.estimator.tol / max(1, nl)
    #println("NewTol: ", compressor.lrf.convergence.estimator.tol)

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
            (!(dirs == [0]) && isassigned(eₜ, parent(testtree(tree), t))) &&
                for eₜₜ in eₜ[parent(testtree(tree), t)]
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
    Fs::Vector{Vector{Int}},
    eₛ::Vector{Vector{Int}},
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

    for level in levels(trialtree(tree))#enumerate(dirfars)
        transferidcs = Int[]
        transfer = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,T}}[]
        trialclusters = collect(LevelIterator(trialtree(tree), level))
        @tasks for s in trialclusters
            @set ntasks = ntasks
            localblocks = Matrix{T}[]
            localdirs = Int[]
            localpivots = Tuple{Vector{Int},Vector{Int}}[]

            dirs = unique(eₛ[s])
            (!(dirs == [0]) && isassigned(eₛ, parent(trialtree(tree), s))) &&
                for eₛₛ in eₛ[parent(trialtree(tree), s)]
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
