


#=using FastBEAST
import FastBEAST.NminClusterTrees.NminTree

function testfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:length(tree.nodes)]
    clusterlink = FastBEAST.cluster_link(tree)
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[1]], far[2])
        end
    end

    for level in clusterlink[2:end]
        for cluster in level
            append!(sortedfars[cluster], sortedfars[ClusterTrees.parent(tree, cluster)])
        end
    end

    return sortedfars
end

function trialfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:length(tree.nodes)]
    clusterlink = FastBEAST.cluster_link(tree)
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[2]], far[1])
        end
    end

    for level in clusterlink[2:end]
        for cluster in level
            append!(sortedfars[cluster], sortedfars[ClusterTrees.parent(tree, cluster)])
        end
    end

    return sortedfars
end

function findrepresentor(
    tree::NminTree{D}, node::Int, pivots::Vector{Tuple{Vector{Int},Vector{Int}}}, rc::Int
) where {D}
    if pivots[node][rc] != []
        if ClusterTrees.haschildren(tree, node)
            idcs = Int[]
            for child in children(tree, node)
                append!(idcs, findrepresentor(tree, child, pivots, rc))
            end
            return idcs
        else
            return pivots[node][rc]
        end
    else
        if ClusterTrees.haschildren(tree, node)
            idcs = Int[]
            for child in children(tree, node)
                append!(idcs, findrepresentor(tree, child, pivots, rc))
            end
            return idcs
        else
            return value(tree, node)
        end
    end
end

function compress_testtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::ButtomUpCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    buffer=allocate_buttomupbuffer(
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

    sortedfars = testfarfield(test_tree, fars)
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
    for level in reverse(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        _foreach(level) do node
            if sortedfars[node] != []
                pivots[node] = compressor(
                    test_tree,
                    cbuffer,
                    rbuffer,
                    farassembler,
                    node,
                    sortedfars[node],
                    pivots;
                    tol=tol / admlevel,
                    maxrank=maxrank,
                )
            end
        end
        build_testbases!(
            translations,
            translationidcs,
            testmoments,
            testmomentidcs,
            cbuffer,
            pivots,
            level,
            test_tree;
            multithreading=multithreading,
        )
        pushfirst!(leveledtranslations, Dict(translationidcs .=> translations))
    end

    return Dict(testmomentidcs .=> testmoments), leveledtranslations, pivots
end

function compress_trialtree(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::ButtomUpCompressor,
    ::Type{K};
    maxrank=40,
    tol=1e-4,
    buffer=allocate_buttomupbuffer(
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

    sortedfars = trialfarfield(trial_tree, fars)
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
    for level in reverse(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]

        _foreach(level) do node
            if sortedfars[node] != []
                pivots[node] = compressor(
                    trial_tree,
                    cbuffer,
                    rbuffer,
                    farassembler,
                    sortedfars[node],
                    node,
                    pivots;
                    tol=tol / admlevel,
                    maxrank=maxrank,
                )
            end
        end
        build_trialbases!(
            translations,
            translationidcs,
            trialmoments,
            trialmomentidcs,
            rbuffer,
            pivots,
            level,
            trial_tree;
            multithreading=multithreading,
        )
        pushfirst!(leveledtranslations, Dict(translationidcs .=> translations))
    end

    return Dict(trialmomentidcs .=> trialmoments), leveledtranslations, pivots
end
=#
