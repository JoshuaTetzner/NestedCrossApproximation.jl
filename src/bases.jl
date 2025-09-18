
# directional topdowncompressor
function build_testbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    testclusters::Vector{Int},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    dtree,
    tree;
    ntasks=1,
) where {I,K}
    lk = Threads.SpinLock()

    for node in testclusters
        #ntasks = ntasks
        if isleaf(dtree, level)
            if isassigned(pivots, node)
                localbases = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[node]
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in piv[1]
                    ]
                    U = blocks[node][dir] / blocks[node][dir][rows, :]
                    push!(
                        localbases,
                        H2BasisBlock(U, H2Trees.values(tree, node), piv[2], Int[]),
                    )
                end
                lock(lk) do
                    push!(basesidcs, node)
                    push!(bases, Dict(keys(pivots[node]) .=> localbases))
                end
            end
        end
    end

    level == 1 && return nothing

    parentclusters = collect(H2Trees.LevelIterator(tree, level - 1))
    for node in parentclusters
        #ntasks = ntasks

        if !iszero(H2Trees.firstchild(tree, node)) && isassigned(pivots, node)#] != ([], [])
            transferdirs = Int[]
            dirtransfers = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[node]
                transferblocks = Matrix{K}[]
                children = collect(H2Trees.ChildIterator(tree, node))
                push!(transferdirs, dir)
                for child in children
                    if !haskey(pivots[child], parent(dtree, dir))
                        println(
                            "child: ",
                            child,
                            ", dir: ",
                            dir,
                            ", options: ",
                            keys(pivots[child]),
                        )
                    end
                    crows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in pivots[child][parent(dtree, dir)][1]
                    ]
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in piv[1]
                    ]
                    Θ = blocks[node][dir][crows, :] / blocks[node][dir][rows, :]

                    push!(transferblocks, Θ)
                end
                push!(dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children))
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(transfer, Dict(transferdirs .=> dirtransfers))
            end
        end
    end
end

# topdowncompressor
function build_testbases!(
    transfer::Vector{H2BasisBlock{I,K}},
    transferidcs::Vector{Int},
    bases::Vector{H2BasisBlock{I,K}},
    basesidcs::Vector{Int},
    buffer::Tuple{Matrix{K},Matrix{K}},
    testclusters::Vector{Int},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
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
            transferblocks = Matrix{K}[]
            children = collect(H2Trees.ChildIterator(tree, node))
            for child in children
                Θ =
                    buffer[3 - idx][pivots[child][1], 1:length(pivots[node][2])] /
                    buffer[3 - idx][pivots[node][1], 1:length(pivots[node][2])]
                push!(transferblocks, Θ)
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(
                    transfer,
                    H2BasisBlock(
                        transferblocks, pivots[node][1], pivots[node][2], children
                    ),
                )
            end
        end
    end
end

# directional trialbases
function build_trialbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    trialclusters::Vector{Int},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    dtree,
    tree;
    ntasks=1,
) where {I,K}
    lk = Threads.SpinLock()

    for node in trialclusters
        #ntasks = ntasks
        if isleaf(dtree, level)
            if isassigned(pivots, node)
                localbases = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[node]
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in piv[2]
                    ]

                    V = blocks[node][dir][:, cols] \ blocks[node][dir]
                    push!(
                        localbases,
                        H2BasisBlock(V, piv[1], H2Trees.values(tree, node), Int[]),
                    )
                end
                lock(lk) do
                    push!(basesidcs, node)
                    push!(bases, Dict(keys(pivots[node]) .=> localbases))
                end
            end
        end
    end

    level == 1 && return nothing

    parentclusters = collect(H2Trees.LevelIterator(tree, level - 1))
    for node in parentclusters
        #ntasks = ntasks

        if !iszero(H2Trees.firstchild(tree, node)) && isassigned(pivots, node)#] != ([], [])
            transferdirs = Int[]
            dirtransfers = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[node]
                transferblocks = Matrix{K}[]
                children = collect(H2Trees.ChildIterator(tree, node))
                push!(transferdirs, dir)
                for child in children
                    ccols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in pivots[child][parent(dtree, dir)][2]
                    ]
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, node)) for
                        idx in piv[2]
                    ]
                    Θ = blocks[node][dir][:, cols] \ blocks[node][dir][:, ccols]

                    push!(transferblocks, Θ)
                end
                push!(dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children))
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(transfer, Dict(transferdirs .=> dirtransfers))
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
            transferblocks = Matrix{K}[]
            children = collect(H2Trees.ChildIterator(tree, node))
            for child in children
                Θ =
                    buffer[3 - idx][1:length(pivots[node][1]), pivots[node][2]] \
                    buffer[3 - idx][1:length(pivots[node][1]), pivots[child][2]]
                push!(transferblocks, Θ)
            end
            lock(lk) do
                push!(transferidcs, node)
                push!(
                    transfer,
                    H2BasisBlock(
                        transferblocks, pivots[node][1], pivots[node][2], children
                    ),
                )
            end
        end
    end
end
