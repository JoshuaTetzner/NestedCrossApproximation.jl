
# topdowncompressor
function build_testbases!(
    transfer::Vector{H2BasisBlock{I,K}},
    transferidcs::Vector{Int},
    bases::Vector{H2BasisBlock{I,K}},
    basesidcs::Vector{Int},
    buffer::Tuple{Matrix{K},Matrix{K}},
    testclusters::Vector{Int},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    level::Int,
    tree;
    ntasks=1,
) where {I,K}
    lk = Threads.SpinLock()
    idx = bufferidx(level)

    @tasks for node in testclusters
        @set ntasks = ntasks
        if iszero(H2Trees.firstchild(tree, node))
            if isassigned(pivots, node)
                U =
                    buffer[idx][H2Trees.values(tree, node), 1:length(pivots[node][2])] /
                    buffer[idx][pivots[node][1], 1:length(pivots[node][2])]
                lock(lk) do
                    push!(basesidcs, node)
                    push!(
                        bases,
                        H2BasisBlock(U, H2Trees.values(tree, node), pivots[node][2], Int[]),
                    )
                end
            end
        end
    end

    level == 1 && return nothing

    parentclusters = collect(H2Trees.LevelIterator(tree, level - 1))
    @tasks for node in parentclusters
        @set ntasks = ntasks
        if !iszero(H2Trees.firstchild(tree, node)) && isassigned(pivots, node)#] != ([], [])
            translationblocks = Matrix{K}[]
            children = collect(H2Trees.ChildIterator(tree, node))
            for child in children
                Θ =
                    buffer[3 - idx][pivots[child][1], 1:length(pivots[node][2])] /
                    buffer[3 - idx][pivots[node][1], 1:length(pivots[node][2])]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(
                    transfer,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], children
                    ),
                )
            end
        end
    end
end

function build_trialbases!(
    transfer::Vector{H2BasisBlock{I,K}},
    transferidcs::Vector{Int},
    bases::Vector{H2BasisBlock{I,K}},
    basesidcs::Vector{Int},
    buffer::Tuple{Matrix{K},Matrix{K}},
    trialclusters::Vector{Int},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    level::Int,
    tree;
    ntasks=1,
) where {I,K}
    lk = Threads.SpinLock()
    idx = bufferidx(level)

    @tasks for node in trialclusters
        @set ntasks = ntasks
        if iszero(H2Trees.firstchild(tree, node))
            if isassigned(pivots, node)
                V =
                    buffer[idx][1:length(pivots[node][1]), pivots[node][2]] \
                    buffer[idx][1:length(pivots[node][1]), H2Trees.values(tree, node)]
                lock(lk) do
                    push!(basesidcs, node)
                    push!(
                        bases,
                        H2BasisBlock(V, pivots[node][1], H2Trees.values(tree, node), Int[]),
                    )
                end
            end
        end
    end

    level == 1 && return nothing

    parentclusters = collect(H2Trees.LevelIterator(tree, level - 1))
    @tasks for node in parentclusters
        @set ntasks = ntasks
        if !iszero(H2Trees.firstchild(tree, node)) && isassigned(pivots, node)#] != ([], [])
            translationblocks = Matrix{K}[]
            children = collect(H2Trees.ChildIterator(tree, node))
            for child in children
                Θ =
                    buffer[3 - idx][1:length(pivots[node][1]), pivots[node][2]] \
                    buffer[3 - idx][1:length(pivots[node][1]), pivots[child][2]]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(
                    transfer,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], children
                    ),
                )
            end
        end
    end
end
