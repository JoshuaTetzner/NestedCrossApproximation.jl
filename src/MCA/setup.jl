using FastBEAST
import FastBEAST.NminClusterTrees.NminTree

function testfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:length(tree.nodes)]
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[1]], far[2])
            append!(sortedfars[far[1]], sortedfars[ClusterTrees.parent(tree, far[1])])
        end
    end

    return sortedfars
end

function trialfarfield(tree, levelfars::Vector{Vector{Tuple{Int,Int}}})
    sortedfars = [Int[] for i in 1:length(tree.nodes)]
    for fars in levelfars
        for far in fars
            push!(sortedfars[far[2]], far[1])
            append!(sortedfars[far[1]], sortedfars[ClusterTrees.parent(tree, far[1])])
        end
    end

    return sortedfars
end

function first_testpivot(
    test_tree,
    trial_tree,
    test_farfield,
    test_buffer,
    testpivots,
    tnode,
    farassembler,
    testpivoting,
    trialpivoting,
    ::Type{K},
) where {K}
    testidcs = value(test_tree, tnode)
    trialidcs = value(trial_tree, test_farfield[tnode])
    lm = FastBEAST.LRF.LazyMatrix(farassembler, testidcs, trialidcs, K)
    pivoting = testpivoting(
        trialidcs; ref=sum(trialpivoting.pos[testidcs]) / length(testidcs)
    )
    push!(testpivots[tnode][1], pivoting())
    @views lm.μ(
        test_buffer[testidcs, 1:1],
        lm.τ[1:length(testidcs)],
        lm.σ[testpivots[tnode][1][1]:testpivots[tnode][1][1]],
    )
    push!(testpivots[tnode][2], argmax(abs.(test_buffer[testidcs, 1:1])))

    return pivoting
end

function first_trialpivot(
    test_tree,
    trial_tree,
    trial_farfield,
    trial_buffer,
    trialpivots,
    tnode,
    farassembler,
    testpivoting,
    trialpivoting,
    ::Type{K},
) where {K}
    testidcs = value(test_tree, trial_farfield[tnode])
    trialidcs = value(trial_tree, tnode)
    lm = FastBEAST.LRF.LazyMatrix(farassembler, testidcs, trialidcs, K)
    pivoting = trialpivoting(
        testidcs; ref=sum(testpivoting.pos[trialidcs]) / length(trialidcs)
    )
    push!(trialpivots[tnode][2], pivoting())
    @views lm.μ(
        trial_buffer[1:1, trialidcs],
        lm.τ[trialpivots[tnode][2][1]:trialpivots[tnode][2][1]],
        lm.σ[1:length(trialidcs)],
    )
    push!(trialpivots[tnode][2], argmax(abs.(trial_buffer[1:1, trialidcs])))

    return pivoting
end

function addtestpivot!(
    test_tree, test_buffer, tnode, farassembler, testpivstrats, npivot, ::Type{K}
) where {K}
    testidcs = value(test_tree, tnode)
    trialidcs = value(trial_tree, test_farfield[tnode])
    lm = FastBEAST.LRF.LazyMatrix(farassembler, testidcs, trialidcs, K)
    push!(testpivots[tnode][1], testpivstrats[node](npivot))
    @views lm.μ(
        test_buffer[testidcs, npivot:npivot],
        lm.τ[1:length(testidcs)],
        lm.σ[testpivots[tnode][1][npivot]:testpivots[tnode][1][npivot]],
    )

    return push!(testpivots[tnode][2], argmax(abs.(test_buffer[testidcs, 1:1])))
end

function buildbytestindex!(
    buffer::Matrix{K},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    indices,
    tree;
    multithreading=true,
) where {I,K}
    momentidcs = Int[]
    moments = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    translationidcs = Int[]
    translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
    lk = Threads.SpinLock()

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(indices) do (node)
        if !ClusterTrees.haschildren(tree, node)
            if pivots[node] != ([], [])
                U =
                    buffer[value(tree, node), 1:length(pivots[node][2])] /
                    buffer[pivots[node][1], 1:length(pivots[node][2])]
                lock(lk) do
                    push!(momentidcs, node)
                    push!(
                        moments, H2BasisBlock(U, value(tree, node), pivots[node][2], Int[])
                    )
                end
            end
        else
            translationblocks = Matrix{K}[]
            childs = collect(children(tree, node))
            for child in childs
                Θ =
                    buffer[pivots[child][1], 1:length(pivots[node][2])] /
                    buffer[pivots[node][1], 1:length(pivots[node][2])]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(translationidcs, node)
                push!(
                    translations,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], childs
                    ),
                )
            end
        end
    end{
        D,
        F,
    }

    return momentidcs, moments, translationidcs, translations
end

function setup(
    test_tree::NminTree{D},
    trial_tree::NminTree{D},
    farassembler::Function,
    fars::Vector{Vector{Tuple{Int,Int}}},
    #testpivoting::IACAPivoting,
    #trialpivoting::IACAPivoting,
    ::Type{K};
    maxrank=40,
) where {D,K}
    lk = Threads.SpinLock()
    test_farfield = testfarfield(test_tree, fars)
    trial_farfield = trialfarfield(trial_tree, fars)
    test_clusterlink = FastBEAST.cluster_link(test_tree)
    trial_clusterlink = FastBEAST.cluster_link(test_tree)
    @assert length(test_clusterlink) == length(trial_clusterlink)
    nlevel = length(test_clusterlink)
    test_buffer = [zeros(K, test_tree.num_elements, maxrank) for l in 1:nlevel]
    trial_buffer = [zeros(K, maxrank, trial_tree.num_elements) for l in 1:nlevel]
    testpivots = [([], []) for i in eachindex(test_tree.nodes)]
    trialpivots = [([], []) for i in eachindex(trial_tree.nodes)]
    testpivstrats = IACAPivoting{3,Float64}[]
    testpividcs = Int[]

    for level in reverse(1:nlevel)
        for tnode in test_clusterlink[level]
            if test_farfield[tnode] != []
                pivoting = first_testpivot(
                    test_tree,
                    trial_tree,
                    test_farfield,
                    test_buffer[level],
                    testpivots,
                    tnode,
                    farassembler,
                    testpivoting,
                    trialpivoting,
                    K,
                )
                lock(lk) do
                    push!(testpividcs, tnode)
                    push!(testpivstrats, pivoting)
                end
            end
        end
        for tnode in trial_clusterlink[level]
            if trial_farfield[tnode] != []
                pivoting = first_trialpivot(
                    test_tree,
                    trial_tree,
                    trial_farfield,
                    trial_buffer[level],
                    trialpivots,
                    tnode,
                    farassembler,
                    testpivoting,
                    trialpivoting,
                    K,
                )

                lock(lk) do
                    push!(testpividcs, tnode)
                    push!(testpivstrats, pivoting)
                end
            end
        end

        testmomentidcs, testmoments, testtranslationidcs, testtranslations = buildbytestindex!(
            test_buffer[level], testpivots, testpividcs, test_tree
        )
        levelcoupling = assemble_couplingmatrices(
            farassembler, K, fars[level:level], testpivots, trialpivots
        )
    end

    return x = 0
end
