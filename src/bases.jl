
function nestedtestbasis!(
    localbases::Vector{Matrix{K}},
    buffer::Matrix{K},
    tree,
    t::Int,
    pivots::Tuple{Vector{I},Vector{I}},
) where {I,K}
    @views U =
        buffer[H2Trees.values(testtree(tree), t), 1:length(pivots[1])] /
        buffer[pivots[1], 1:length(pivots[1])]
    return push!(localbases, U)
end

# directional topdowncompressor
function build_testbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    data::DirectionalData,
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for t in collect(LevelIterator(tree, level))
        @set ntasks = ntasks
        if isassigned(pivots, t)
            localbases = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[t]
                if isroot(data, tree, node)
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for idx in piv[1]
                    ]
                    U = blocks[t][dir] / blocks[t][dir][rows, :]
                    push!(
                        localbases, H2BasisBlock(U, H2Trees.values(tree, t), piv[2], Int[])
                    )
                end
            end
            if localbases != []
                lock(lk) do
                    push!(basesidcs, t)
                    push!(bases, Dict(keys(pivots[t]) .=> localbases))
                end
            end
        end
    end

    level == 1 && return nothing

    @tasks for t in collect(H2Trees.LevelIterator(tree, level - 1))
        @set ntasks = ntasks
        if !isroot(data, tree, t) && isassigned(pivots, t)
            transferdirs = Int[]
            dirtransfers = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[t]
                transferblocks = Matrix{K}[]
                children = collect(ChildIterator(tree, t))
                push!(transferdirs, dir)
                for child in children
                    crows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                        idx in pivots[child][data.Emap[idx]][1]
                    ]
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for idx in piv[1]
                    ]
                    Θ = blocks[t][dir][crows, :] / blocks[t][dir][rows, :]

                    push!(transferblocks, Θ)
                end
                push!(dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children))
            end
            lock(lk) do
                push!(transferidcs, t)
                push!(transfer, Dict(transferdirs .=> dirtransfers))
            end
        end
    end
end

# directional bottomupcompressor
function build_butestbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    dtree,
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for t in collect(LevelIterator(tree, level))
        @set ntasks = ntasks
        if isassigned(pivots, t)
            if iszero(firstchild(tree, t)) || isroot(dtree, level)
                localbases = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[t]
                    if isroot(dtree, level) || iszero(firstchild(tree, t))
                        rows = [
                            findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                            idx in piv[1]
                        ]
                        if rank(blocks[t][dir][rows, :]) < length(rows)
                            println(piv[1], piv[2])
                            error()
                        end
                        U = blocks[t][dir] / blocks[t][dir][rows, :]
                        push!(
                            localbases,
                            H2BasisBlock(U, H2Trees.values(tree, t), piv[2], Int[]),
                        )
                    end
                end
                if localbases != []
                    lock(lk) do
                        push!(basesidcs, t)
                        push!(bases, Dict(keys(pivots[t]) .=> localbases))
                    end
                end
            else
                transferdirs = Int[]
                dirtransfers = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[t]
                    transferblocks = Matrix{K}[]
                    children = collect(ChildIterator(tree, t))
                    push!(transferdirs, dir)
                    for child in children
                        crows = [
                            findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                            idx in pivots[child][parent(dtree, dir)][1]
                        ]
                        rows = [
                            findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                            idx in piv[1]
                        ]
                        Θ = blocks[t][dir][crows, :] / blocks[t][dir][rows, :]

                        push!(transferblocks, Θ)
                    end
                    push!(
                        dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children)
                    )
                end
                lock(lk) do
                    push!(transferidcs, t)
                    push!(transfer, Dict(transferdirs .=> dirtransfers))
                end
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
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    level::Int,
    tree;
    ntasks=Threads.nthreads(),
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

# bottomupcompressor
function build_testbases!(
    transfer::Vector{H2BasisBlock{I,K}},
    transferidcs::Vector{Int},
    bases::Vector{H2BasisBlock{I,K}},
    basesidcs::Vector{Int},
    buffer::Matrix{K},
    testclusters::Vector{Int},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for node in testclusters
        @set ntasks = ntasks
        if iszero(H2Trees.firstchild(tree, node))
            if isassigned(pivots, node)
                U =
                    buffer[H2Trees.values(tree, node), 1:length(pivots[node][2])] /
                    buffer[pivots[node][1], 1:length(pivots[node][2])]
                lock(lk) do
                    push!(basesidcs, node)
                    push!(
                        bases,
                        H2BasisBlock(U, H2Trees.values(tree, node), pivots[node][2], Int[]),
                    )
                end
            end
        else
            if isassigned(pivots, node)
                transferblocks = Matrix{K}[]
                children = collect(H2Trees.ChildIterator(tree, node))
                for child in children
                    Θ =
                        buffer[pivots[child][1], 1:length(pivots[node][2])] /
                        buffer[pivots[node][1], 1:length(pivots[node][2])]
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
end

function nestedtrialbasis!(
    localbases::Vector{Matrix{K}},
    buffer::Matrix{K},
    tree,
    s::Int,
    pivots::Tuple{Vector{I},Vector{I}},
) where {I,K}
    @views V =
        buffer[1:length(pivots[1]), pivots[2]] \
        buffer[1:length(pivots[1]), H2Trees.values(trialtree(tree), s)]
    return push!(localbases, V)
end

# directional topdowncompressor
function build_trialbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    dtree,
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for s in collect(LevelIterator(tree, level))
        @set ntasks = ntasks
        if isassigned(pivots, s)
            localbases = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[s]
                if isroot(dtree, level) || iszero(firstchild(tree, s))
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for idx in piv[2]
                    ]

                    V = blocks[s][dir][:, cols] \ blocks[s][dir]
                    push!(
                        localbases, H2BasisBlock(V, piv[1], H2Trees.values(tree, s), Int[])
                    )
                end
            end
            if localbases != []
                lock(lk) do
                    push!(basesidcs, s)
                    push!(bases, Dict(keys(pivots[s]) .=> localbases))
                end
            end
        end
    end

    level == 1 && return nothing

    @tasks for s in collect(H2Trees.LevelIterator(tree, level - 1))
        @set ntasks = ntasks
        if !iszero(firstchild(tree, s)) &&
            !isroot(dtree, level - 1) &&
            isassigned(pivots, s)
            transferdirs = Int[]
            dirtransfers = H2BasisBlock{I,K}[]
            for (dir, piv) in pivots[s]
                transferblocks = Matrix{K}[]
                children = collect(H2Trees.ChildIterator(tree, s))
                push!(transferdirs, dir)
                for child in children
                    ccols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                        idx in pivots[child][parent(dtree, dir)][2]
                    ]
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for idx in piv[2]
                    ]
                    Θ = blocks[s][dir][:, cols] \ blocks[s][dir][:, ccols]

                    push!(transferblocks, Θ)
                end
                push!(dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children))
            end
            lock(lk) do
                push!(transferidcs, s)
                push!(transfer, Dict(transferdirs .=> dirtransfers))
            end
        end
    end
end

#directional bottomupcompressor
function build_butrialbases!(
    transfer::Vector{Dict{Int,H2BasisBlock{I,K}}},
    transferidcs::Vector{Int},
    bases::Vector{Dict{Int,H2BasisBlock{I,K}}},
    basesidcs::Vector{Int},
    blocks::Vector{Dict{Int,Matrix{K}}},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    level::Int,
    dtree,
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for s in collect(LevelIterator(tree, level))
        @set ntasks = ntasks
        if isassigned(pivots, s)
            if iszero(firstchild(tree, s)) || isroot(dtree, level)
                localbases = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[s]
                    if isroot(dtree, level) || iszero(firstchild(tree, s))
                        cols = [
                            findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                            idx in piv[2]
                        ]

                        V = blocks[s][dir][:, cols] \ blocks[s][dir]
                        push!(
                            localbases,
                            H2BasisBlock(V, piv[1], H2Trees.values(tree, s), Int[]),
                        )
                    end
                end
                if localbases != []
                    lock(lk) do
                        push!(basesidcs, s)
                        push!(bases, Dict(keys(pivots[s]) .=> localbases))
                    end
                end
            else
                transferdirs = Int[]
                dirtransfers = H2BasisBlock{I,K}[]
                for (dir, piv) in pivots[s]
                    transferblocks = Matrix{K}[]
                    children = collect(H2Trees.ChildIterator(tree, s))
                    push!(transferdirs, dir)
                    for child in children
                        if !isassigned(pivots, child)
                            println("not assigned: ", child, ", ", s)
                        end
                        ccols = [
                            findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                            idx in pivots[child][parent(dtree, dir)][2]
                        ]
                        cols = [
                            findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                            idx in piv[2]
                        ]
                        Θ = blocks[s][dir][:, cols] \ blocks[s][dir][:, ccols]

                        push!(transferblocks, Θ)
                    end
                    push!(
                        dirtransfers, H2BasisBlock(transferblocks, piv[1], piv[2], children)
                    )
                end
                lock(lk) do
                    push!(transferidcs, s)
                    push!(transfer, Dict(transferdirs .=> dirtransfers))
                end
            end
        end
    end
end

# topdowncompressor
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
    ntasks=Threads.nthreads(),
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
        if !iszero(H2Trees.firstchild(tree, node)) && isassigned(pivots, node)
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

# bottomupcompressor
function build_trialbases!(
    transfer::Vector{H2BasisBlock{I,K}},
    transferidcs::Vector{Int},
    bases::Vector{H2BasisBlock{I,K}},
    basesidcs::Vector{Int},
    buffer::Matrix{K},
    trialclusters::Vector{Int},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    tree;
    ntasks=Threads.nthreads(),
) where {I,K}
    lk = Threads.SpinLock()

    @tasks for node in trialclusters
        @set ntasks = ntasks
        if iszero(H2Trees.firstchild(tree, node))
            if isassigned(pivots, node)
                V =
                    buffer[1:length(pivots[node][1]), pivots[node][2]] \
                    buffer[1:length(pivots[node][1]), H2Trees.values(tree, node)]
                lock(lk) do
                    push!(basesidcs, node)
                    push!(
                        bases,
                        H2BasisBlock(V, pivots[node][1], H2Trees.values(tree, node), Int[]),
                    )
                end
            end
        else
            if isassigned(pivots, node)
                transferblocks = Matrix{K}[]
                children = collect(H2Trees.ChildIterator(tree, node))
                for child in children
                    Θ =
                        buffer[1:length(pivots[node][1]), pivots[node][2]] \
                        buffer[1:length(pivots[node][1]), pivots[child][2]]
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
end
