function testfars(nnodes::Int, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:nnodes]
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[1]], far[2])
        end
    end

    return sortedfars
end

function trialfars(nnodes::Int, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:nnodes]
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[2]], far[1])
        end
    end

    return sortedfars
end

function compress_testtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::TopDownCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    buffer=allocate_buffer(
        K,
        channel(compressor, trial_tree.num_elements; maxrank=maxrank),
        buffer(compressor, test_tree.num_elements; maxrank=maxrank);
    ),
    multithreading=true,
) where {D,K}
    testmomentidcs = Int[]
    testmoments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    leveledtranslations = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}[]
    pivots = [(Int[], Int[]) for i in eachindex(test_tree.nodes)]

    sortedfars = testfars(length(test_tree.nodes), fars)
    clusterlink = FastBEAST.cluster_link(test_tree)
    rbuffer, cbuffer = buffer

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    admlevel = 0
    for cl in eachindex(fars)
        if fars[cl] != []
            admlevel += 1
        end
    end
    println("Tolerance: ", tol / admlevel)
    for (levelidx, level) in enumerate(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        iseven(levelidx) ? ridx = 1 : ridx = 2
        _foreach(level) do node
            colidcs = value(trial_tree, sortedfars[node])
            (node != 1) &&
                (colidcs = vcat(colidcs, pivots[ClusterTrees.parent(test_tree, node)][2]))
            if colidcs != []
                rowidcs = value(test_tree, node)
                pivots[node] = compressor(
                    cbuffer[ridx],
                    rbuffer,
                    farassembler,
                    rowidcs,
                    colidcs;
                    tol=tol / admlevel,
                    maxrank=maxrank,
                )
            end
        end
        (levelidx > 1) && build_testbases!(
            translations,
            translationidcs,
            testmoments,
            testmomentidcs,
            cbuffer,
            pivots,
            clusterlink,
            levelidx,
            test_tree;
            multithreading=multithreading,
        )
        push!(leveledtranslations, Dict(translationidcs .=> translations))
    end

    return Dict(testmomentidcs .=> testmoments), leveledtranslations, pivots
end

function compress_trialtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::TopDownCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    buffer=allocate_buffer(
        K,
        reverse(channel(compressor, test_tree.num_elements; maxrank=maxrank)),
        reverse(buffer(compressor, trial_tree.num_elements; maxrank=maxrank));
    ),
    multithreading=true,
) where {D,K}
    trialmomentidcs = Int[]
    trialmoments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    leveledtranslations = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}[]
    pivots = [(Int[], Int[]) for i in eachindex(trial_tree.nodes)]

    sortedfars = trialfars(length(trial_tree.nodes), fars)
    clusterlink = FastBEAST.cluster_link(trial_tree)
    cbuffer, rbuffer = buffer

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    admlevel = 0
    for cl in eachindex(fars)
        if fars[cl] != []
            admlevel += 1
        end
    end
    println("Tolerance: ", tol / admlevel)
    for (levelidx, level) in enumerate(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        iseven(levelidx) ? ridx = 1 : ridx = 2

        _foreach(level) do node
            rowidcs = value(test_tree, sortedfars[node])
            (node != 1) &&
                (rowidcs = vcat(rowidcs, pivots[ClusterTrees.parent(trial_tree, node)][1]))
            if rowidcs != []
                colidcs = value(trial_tree, node)
                pivots[node] = compressor(
                    cbuffer,
                    rbuffer[ridx],
                    farassembler,
                    rowidcs,
                    colidcs;
                    tol=tol / admlevel,
                    maxrank=maxrank,
                )
            end
        end
        (levelidx > 1) && build_trialbases!(
            translations,
            translationidcs,
            trialmoments,
            trialmomentidcs,
            rbuffer,
            pivots,
            clusterlink,
            levelidx,
            trial_tree;
            multithreading=multithreading,
        )
        push!(leveledtranslations, Dict(translationidcs .=> translations))
    end

    return Dict(trialmomentidcs .=> trialmoments), leveledtranslations, pivots
end
