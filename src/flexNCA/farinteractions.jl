abstract type TopDown end
abstract type ButtomUp end

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

function compress(tree, fars, farassembler, compressor::ButtomUp) end

function compress(test_tree, trial_tree, fars, farassembler, compressor::ButtomUp) end

function compress(
    tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    compressor::TopDownCompressor,
    ::Type{K};
    multithreading=false,
    maxrank=40,
) where {D,K}
    sortedfars = testfars(length(tree.nodes), fars)
    clusterlink = FastBEAST.cluster_link(tree)
    momentidcs = Int[]
    moments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    levtranslations = Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}[]

    cbuffer, rbuffer = allocate_sym_buffer(
        K, compressor, tree.num_elements, tree.num_elements
    )

    pivots = [(Int[], Int[]) for i in eachindex(tree.nodes)]

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    for (levelidx, level) in enumerate(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        iseven(levelidx) ? ridx = 1 : ridx = 2
        _foreach(level) do node
            farfield = value(tree, sortedfars[node])
            if node != 1
                farfield = vcat(farfield, pivots[ClusterTrees.parent(tree, node)][2])
            end
            if farfield != []
                localrbuffer = take!(rbuffer)
                idcs = value(tree, node)
                pivots[node] = compressor(
                    cbuffer[ridx], localrbuffer, farassembler, idcs, farfield
                )
                put!(rbuffer, localrbuffer)
            end
        end
        if levelidx > 1
            build_testh2blocks!(
                translations,
                translationidcs,
                moments,
                momentidcs,
                cbuffer,
                pivots,
                clusterlink,
                sortedfars,
                levelidx,
                tree,
            )
        end
        push!(levtranslations, Dict(translationidcs .=> translations))
    end

    return Dict(momentidcs .=> moments), levtranslations, pivots
end

function compress(test_tree, trial_tree, farassembler, fars, compressor::TopDown)
    sortedtestfars = testfars(length(test_tree.nodes), fars)
    testclusterlink = FastBEAST.cluster_link(test_tree)
    sortedtrialfars = trialfars(length(trial_tree.nodes), fars)
    return trialclusterlink = FastBEAST.cluster_link(trial_tree)
end
