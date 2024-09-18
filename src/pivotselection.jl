using Base.Threads
using ClusterTrees
using ProgressMeter
using ThreadsX

function sort_interactions(
    nnodes::Int, levelfars::Vector{Vector{Tuple{Int,Int}}}; testortrial=1
)
    sortedfars = [Int[] for i in 1:nnodes]

    testortrial == 1 ? trialortest = 2 : trialortest = 1
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[testortrial]], far[trialortest])
        end
    end

    return sortedfars
end

function interactionindices(
    tree::ClusterTrees.NminTrees.NminTree{D},
    nodeidx::I,
    interactions::Vector{I},
    inheritedpivots::Vector{I}
) where {I, D}

    index_set = Int[]
    cluster_maps = (zeros(I, length(interactions)+1), interactions)

    for (ind, interaction) in enumerate(interactions)
        append!(index_set, value(tree, interaction))
        cluster_maps[1][ind+1] = length(index_set)
    end
    append!(index_set, inheritedpivots)

    return index_set, cluster_maps
end

function row_pivot_selection(
    test_tree::ClusterTrees.NminTrees.NminTree{D},
    trial_tree::ClusterTrees.NminTrees.NminTree{D},
    fars::Vector{Vector{Tuple{I, I}}},
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    verbose=false,
    multithreading=true,
) where {I, K, D}
    
    clusterblocks = Vector{PivotBlocks{I, K}}(undef, length(test_tree.nodes))
    interactionlist = sort_interactions(
        length(test_tree.nodes), fars; testortrial=1
    )
    clusterlink = FastBEAST.cluster_link(test_tree)
    
    if verbose
        p = Progress(sum(length.(clusterlink)), desc="Computing row pivots: ")
    end
    if compressor isa FastBEAST.ACAOptions
        am = allocate_aca_memory(
            K, 
            length(value(test_tree, 1)), 
            length(value(trial_tree, 1)), 
            multithreading; 
            maxrank=compressor.maxrank, 
        )
    else
        am = allocate_pca_memory_rm(
            K, 
            length(value(test_tree, 1)), 
            length(value(trial_tree, 1)), 
            multithreading; 
            maxrank=compressor.maxrank, 
        )
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for level in clusterlink
        _foreach(level) do (nodeidx) 
            childrange = FastBEAST.child_link(test_tree, nodeidx)
            
            parent = ClusterTrees.parent(test_tree, nodeidx)
            if parent != 0 && isassigned(clusterblocks, parent)
                inheritedpivots = clusterblocks[parent].M.σ[clusterblocks[parent].M.M.σ]
            else
                inheritedpivots = Int[]
            end

            noadm=false
            if interactionlist[nodeidx] == 0.0
                noadm = true
            end

            tindices, tclustermaps = interactionindices(
                trial_tree, nodeidx, interactionlist[nodeidx], inheritedpivots
            )

            if tindices != []
                sindices = value(test_tree, nodeidx)
                
                if compressor isa FastBEAST.ACAOptions
                    clusterblocks[nodeidx] = PivotBlocks(
                        getcompressedmatrixview(
                            matrixassembler,
                            sindices,
                            tindices,
                            K,
                            am[Threads.threadid()],
                            compressor;
                            noadm=noadm
                        ),
                        tclustermaps,
                        childrange
                    )
                else
                    refcenter = test_tree.nodes[nodeidx].node.data.ct
                    clusterblocks[nodeidx] = PivotBlocks(
                        getcompressedmatrix_rm(
                            matrixassembler,
                            sindices,
                            tindices,
                            K,
                            am[Threads.threadid()],
                            compressor;
                            refcenter=refcenter,
                            noadm=noadm
                        ),
                        tclustermaps,
                        childrange
                    )
                end
            end

            verbose && next!(p)
        end
    end

    return clusterblocks
end


function column_pivot_selection(
    test_tree::ClusterTrees.NminTrees.NminTree{D},
    trial_tree::ClusterTrees.NminTrees.NminTree{D},
    fars::Vector{Vector{Tuple{I, I}}},
    matrixassembler,
    ::Type{K};
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    verbose=false,
    multithreading=true,
) where {I, K, D}

    clusterblocks = Vector{PivotBlocks{I, K}}(undef, length(trial_tree.nodes))
    interactionlist = sort_interactions(
        length(trial_tree.nodes), fars; testortrial=2
    )
    clusterlink = FastBEAST.cluster_link(trial_tree)

    if verbose
        p = Progress(sum(length.(clusterlink)), desc="Computing column pivots: ")
    end
    
    if compressor isa FastBEAST.ACAOptions
        am = allocate_aca_memory(
            K, 
            length(value(test_tree, 1)), 
            length(value(trial_tree, 1)), 
            multithreading; 
            maxrank=compressor.maxrank, 
        )
    else
        am = allocate_pca_memory_cm(
            K, 
            length(value(test_tree, 1)), 
            length(value(trial_tree, 1)), 
            multithreading; 
            maxrank=compressor.maxrank, 
        )
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for level in clusterlink
        _foreach(level) do (nodeidx) 
            childrange = FastBEAST.child_link(trial_tree, nodeidx)

            parent = ClusterTrees.parent(trial_tree, nodeidx)
            if parent != 0 && isassigned(clusterblocks, parent)
                inheritedpivots = clusterblocks[parent].M.τ[clusterblocks[parent].M.M.τ]
            else
                inheritedpivots = Int[]
            end

            noadm=false
            if interactionlist[nodeidx] == 0.0
                noadm = true
            end

            sindices, sclustermaps = interactionindices(
                test_tree, nodeidx, interactionlist[nodeidx], inheritedpivots
            )

            if sindices != []
                tindices = value(trial_tree, nodeidx)
                if compressor isa FastBEAST.ACAOptions
                    clusterblocks[nodeidx] = PivotBlocks(
                        getcompressedmatrixview(
                            matrixassembler,
                            sindices,
                            tindices,
                            K,
                            am[Threads.threadid()],
                            compressor;
                            noadm=noadm
                        ),
                        sclustermaps,
                        childrange
                    )
                else
                    refcenter = trial_tree.nodes[nodeidx].node.data.ct
                    clusterblocks[nodeidx] = PivotBlocks(
                        getcompressedmatrix_cm(
                            matrixassembler,
                            sindices,
                            tindices,
                            K,
                            am[Threadsthreadid()],
                            compressor,
                            refcenter=refcenter,
                            noadm=noadm
                        ),
                        sclustermaps,
                        childrange
                    )
                end
            end

            verbose && next!(p)
        end
    end

    return clusterblocks
end

