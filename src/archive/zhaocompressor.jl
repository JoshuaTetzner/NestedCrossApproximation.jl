function countpivots(idcs::Vector{Int}, pivots::Vector{Tuple{Vector{Int},Vector{Int}}})
    npivots = 0
    for idx in idcs
        npivots += length(pivots[idx][1])
    end
    return npivots
end

function findoffspring(
    node::Int, tree::NminTree{D}, bound::Int, pivots::Vector{Tuple{Vector{Int},Vector{Int}}}
) where {D}
    idcs = Int[node]
    temp = Int[]

    while countpivots(idcs, pivots) < bound
        idcs, temp = temp, idcs
        for idx in temp
            if !ClusterTrees.haschildren(tree, idx)
                push!(idcs, idx)
            else
                for child in children(tree, idx)
                    push!(idcs, child)
                end
            end
        end
        if length(idcs) == length(temp)
            break
        end
        temp = Int[]
    end
    return idcs
end

function fixmissingtest!(pivots, cbuffer, tree, clusterlink, assembler; multithreading=true)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (levelidx, level) in enumerate(clusterlink)
        _foreach(level) do node
            if ClusterTrees.haschildren(tree, node)
                if pivots[node] != ([], [])
                    missingp = Int[]
                    for child in children(tree, node)
                        for p in pivots[child][1]
                            if !(p in pivots[node][1])
                                push!(missingp, p)
                            end
                        end
                    end
                    if missingp != []
                        cbuffer[levelidx][missingp, 1:length(pivots[node][2])] .= 0.0
                        @views assembler(
                            cbuffer[levelidx][missingp, 1:length(pivots[node][2])],
                            missingp,
                            pivots[node][2],
                        )
                    end
                end
            end
        end
    end
end

function fixmissingtrial!(
    pivots, rbuffer, tree, clusterlink, assembler; multithreading=true
)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (levelidx, level) in enumerate(clusterlink)
        _foreach(level) do node
            if ClusterTrees.haschildren(tree, node)
                if pivots[node] != ([], [])
                    missingp = Int[]
                    for child in children(tree, node)
                        for p in pivots[child][2]
                            if !(p in pivots[node][2])
                                push!(missingp, p)
                            end
                        end
                    end
                    if missingp != []
                        rbuffer[levelidx][1:length(pivots[node][1]), missingp] .= 0.0
                        @views assembler(
                            rbuffer[levelidx][1:length(pivots[node][1]), missingp],
                            pivots[node][1],
                            missingp,
                        )
                    end
                end
            end
        end
    end
end

function compress_testtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::ZhaoCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    multithreading=true,
) where {D,K}
    sortedfars = testfars(length(test_tree.nodes), fars)
    clusterlink = FastBEAST.cluster_link(test_tree)
    pivots = [(Int[], Int[]) for i in eachindex(test_tree.nodes)]

    cbuffer = [zeros(K, test_tree.num_elements, maxrank) for l in clusterlink]
    rbuffer = Channel{Matrix{K}}(Threads.nthreads())
    for task in 1:Threads.nthreads()
        put!(rbuffer, zeros(K, maxrank, trial_tree.num_elements))
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    println("ButtomUp")
    for (levelidx, level) in enumerate(reverse(clusterlink))
        _foreach(level) do node
            rowidcs = Int[]
            colidcs = Int[]

            if !ClusterTrees.haschildren(test_tree, node)
                rowidcs = value(test_tree, node)
                colidcs = value(trial_tree, sortedfars[node])
            else
                for child in children(test_tree, node)
                    if sortedfars[child] == []
                        append!(rowidcs, value(test_tree, child))
                    else
                        append!(rowidcs, pivots[child][1])
                    end
                end
                #rowidcs = value(test_tree, node)
                #rowidcs = value(test_tree, node)

                for admnode in sortedfars[node]
                    if ClusterTrees.haschildren(trial_tree, admnode)
                        for child in children(trial_tree, admnode)
                            if sortedfars[child] == []
                                append!(colidcs, value(trial_tree, child))
                            else
                                append!(colidcs, pivots[child][1])
                            end
                        end
                    else
                        append!(colidcs, value(trial_tree, admnode))
                    end
                end
            end

            if colidcs != [] && rowidcs != []
                pivots[node] = compressor(
                    cbuffer[end - (levelidx - 1)],
                    rbuffer,
                    farassembler,
                    unique(rowidcs),
                    unique(colidcs);
                    tol=tol,
                    maxrank=maxrank,
                )
            end
        end
    end
    println("TopDown")
    for (levelidx, level) in enumerate(clusterlink)
        _foreach(level) do node
            if ClusterTrees.parent(test_tree, node) > 0
                if pivots[ClusterTrees.parent(test_tree, node)][2] != Int[]
                    colidcs = vcat(
                        pivots[node][2], pivots[ClusterTrees.parent(test_tree, node)][2]
                    )
                    rowidcs = Int[]
                    bound = length(colidcs)
                    if !ClusterTrees.haschildren(test_tree, node) ||
                        length(value(test_tree, node)) < bound
                        rowidcs = value(test_tree, node)
                    else
                        nodeidcs = findoffspring(node, test_tree, bound, pivots)
                        for nodeidx in nodeidcs
                            if sortedfars[nodeidx] == []
                                append!(rowidcs, value(test_tree, nodeidx))
                            else
                                append!(rowidcs, pivots[nodeidx][1])
                            end
                        end
                    end
                    #rowidcs = value(test_tree, node)

                    if colidcs != [] && rowidcs != []
                        #cbuffer[levelidx][value(test_tree, node), 1:length(pivots[node][2])] .= K(
                        #    0.0
                        #)
                        pivots[node] = compressor(
                            cbuffer[levelidx],
                            rbuffer,
                            farassembler,
                            unique(rowidcs),
                            unique(colidcs);
                            tol=tol,
                            maxrank=maxrank,
                        )
                    end
                end
            end
        end
    end

    testmomentidcs = Int[]
    testmoments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    leveledtranslations = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}[]
    fixmissingtest!(
        pivots, cbuffer, test_tree, clusterlink, farassembler; multithreading=multithreading
    )
    build_testbases!(
        leveledtranslations,
        testmoments,
        testmomentidcs,
        cbuffer,
        pivots,
        clusterlink,
        test_tree;
        multithreading=multithreading,
    )

    return Dict(testmomentidcs .=> testmoments), leveledtranslations, pivots
end

function compress_trialtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::ZhaoCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    multithreading=true,
) where {D,K}
    sortedfars = trialfars(length(trial_tree.nodes), fars)
    clusterlink = FastBEAST.cluster_link(trial_tree)
    pivots = [(Int[], Int[]) for i in eachindex(trial_tree.nodes)]

    rbuffer = [zeros(K, maxrank, trial_tree.num_elements) for l in clusterlink]
    cbuffer = Channel{Matrix{K}}(Threads.nthreads())
    for task in 1:Threads.nthreads()
        put!(cbuffer, zeros(K, test_tree.num_elements, maxrank))
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    println("ButtomUp")
    for (levelidx, level) in enumerate(reverse(clusterlink))
        _foreach(level) do node
            rowidcs = Int[]
            colidcs = Int[]

            if !ClusterTrees.haschildren(trial_tree, node)
                rowidcs = value(test_tree, sortedfars[node])
                colidcs = value(trial_tree, node)
            else
                for child in children(trial_tree, node)
                    if pivots[child][2] == []
                        append!(colidcs, value(trial_tree, child))
                    else
                        append!(colidcs, pivots[child][2])
                    end
                end
                #colidcs = value(trial_tree, node)
                for admnode in sortedfars[node]
                    if ClusterTrees.haschildren(test_tree, admnode)
                        for child in children(test_tree, admnode)
                            if sortedfars[child] == []
                                append!(rowidcs, value(trial_tree, child))
                            else
                                append!(rowidcs, pivots[child][2])
                            end
                        end
                    else
                        append!(rowidcs, value(test_tree, admnode))
                    end
                end
            end

            if colidcs != [] && rowidcs != []
                pivots[node] = compressor(
                    cbuffer,
                    rbuffer[end - (levelidx - 1)],
                    farassembler,
                    unique(rowidcs),
                    unique(colidcs);
                    tol=tol,
                    maxrank=maxrank,
                )
            end
        end
    end
    println("TopDown")
    for (levelidx, level) in enumerate(clusterlink)
        _foreach(level) do node
            if ClusterTrees.parent(trial_tree, node) > 0
                if pivots[ClusterTrees.parent(trial_tree, node)][1] != Int[]
                    rowidcs = vcat(
                        pivots[node][1], pivots[ClusterTrees.parent(trial_tree, node)][1]
                    )
                    colidcs = Int[]
                    #colidcs = value(trial_tree, node)

                    bound = length(rowidcs)
                    if !ClusterTrees.haschildren(trial_tree, node) ||
                        length(value(trial_tree, node)) < bound
                        colidcs = value(trial_tree, node)
                    else
                        nodeidcs = findoffspring(node, trial_tree, bound, pivots)
                        for nodeidx in nodeidcs
                            if pivots[nodeidx][2] == []
                                append!(colidcs, value(trial_tree, nodeidx))
                            else
                                append!(colidcs, pivots[nodeidx][2])
                            end
                        end
                    end

                    if colidcs != [] && rowidcs != []
                        #rbuffer[levelidx][1:length(pivots[node][1]), value(trial_tree, node)] .= K(
                        #    0.0
                        #)
                        pivots[node] = compressor(
                            cbuffer,
                            rbuffer[levelidx],
                            farassembler,
                            unique(rowidcs),
                            unique(colidcs);
                            tol=tol,
                            maxrank=maxrank,
                        )
                    end
                end
            end
        end
    end

    testmomentidcs = Int[]
    testmoments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    leveledtranslations = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}[]
    fixmissingtrial!(
        pivots,
        rbuffer,
        trial_tree,
        clusterlink,
        farassembler;
        multithreading=multithreading,
    )
    build_trialbases!(
        leveledtranslations,
        testmoments,
        testmomentidcs,
        rbuffer,
        pivots,
        clusterlink,
        test_tree;
        multithreading=multithreading,
    )

    return Dict(testmomentidcs .=> testmoments), leveledtranslations, pivots
end
