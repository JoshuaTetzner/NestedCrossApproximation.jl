function build_testbases!(
    translations::Vector{H2BasisBlock{I,K}},
    translationidcs::Vector{Int},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    cbuffer::Tuple{Matrix{K},Matrix{K}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    level,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()
    iseven(level) ? idx = 1 : idx = 2

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[level]) do node
        if !ClusterTrees.haschildren(tree, node)
            if pivots[node] != ([], [])
                U =
                    cbuffer[idx][value(tree, node), 1:length(pivots[node][2])] /
                    cbuffer[idx][pivots[node][1], 1:length(pivots[node][2])]
                lock(lk) do
                    push!(momentidcs, node)
                    push!(
                        moments, H2BasisBlock(U, value(tree, node), pivots[node][2], Int[])
                    )
                end
            end
        end
    end
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[level - 1]) do node
        if ClusterTrees.haschildren(tree, node) && pivots[node] != ([], [])
            translationblocks = Matrix{K}[]
            childs = collect(children(tree, node))
            for child in childs
                Θ =
                    cbuffer[3 - idx][pivots[child][1], 1:length(pivots[node][2])] /
                    cbuffer[3 - idx][pivots[node][1], 1:length(pivots[node][2])]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(translationidcs, node)
                push!(
                    translations,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], Int.(childs)
                    ),
                )
            end
        end
    end
end

function build_testbases!(
    translations::Vector{H2BasisBlock{I,K}},
    translationidcs::Vector{Int},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    cbuffer::Matrix{K},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    level,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(level) do node
        if !ClusterTrees.haschildren(tree, node) && pivots[node] != ([], [])
            U =
                cbuffer[value(tree, node), 1:length(pivots[node][2])] /
                cbuffer[pivots[node][1], 1:length(pivots[node][2])]
            lock(lk) do
                push!(momentidcs, node)
                push!(moments, H2BasisBlock(U, value(tree, node), pivots[node][2], Int[]))
            end
        elseif pivots[node] != ([], [])
            translationblocks = Matrix{K}[]
            childs = collect(children(tree, node))
            for child in childs
                Θ =
                    cbuffer[pivots[child][1], 1:length(pivots[node][2])] /
                    cbuffer[pivots[node][1], 1:length(pivots[node][2])]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(translationidcs, node)
                push!(
                    translations,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], Int.(childs)
                    ),
                )
            end
        end
    end
end

function build_trialbases!(
    translations::Vector{H2BasisBlock{I,K}},
    translationidcs::Vector{Int},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    rbuffer::Tuple{Matrix{K},Matrix{K}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    level,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()
    iseven(level) ? idx = 1 : idx = 2

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[level]) do node
        if !ClusterTrees.haschildren(tree, node)
            if pivots[node] != ([], [])
                for i in 1:length(pivots[node][1])
                    if norm(rbuffer[idx][i, pivots[node][2]]) == 0
                        println("zerooooo: ", i, ", ", length(pivots[node][1]))
                    end
                end
                for (x, i) in enumerate(pivots[node][2])
                    if norm(rbuffer[idx][1:length(pivots[node][1]), i]) == 0
                        println("zerooooo2: ", x, ", ", length(pivots[node][1]))
                        println(pivots[node][1], pivots[node][2])
                    end
                end
                V =
                    rbuffer[idx][1:length(pivots[node][1]), pivots[node][2]] \
                    rbuffer[idx][1:length(pivots[node][1]), value(tree, node)]

                lock(lk) do
                    push!(momentidcs, node)
                    push!(
                        moments, H2BasisBlock(V, pivots[node][1], value(tree, node), Int[])
                    )
                end
            end
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[level - 1]) do node
        if ClusterTrees.haschildren(tree, node) && pivots[node] != ([], [])
            translationblocks = Matrix{K}[]
            childs = collect(children(tree, node))
            for child in childs
                if !any(
                    isfinite, rbuffer[3 - idx][1:length(pivots[node][1]), pivots[child][2]]
                )
                    println(rbuffer[3 - idx][1:length(pivots[node][1]), pivots[child][2]])
                end
                Θ =
                    rbuffer[3 - idx][1:length(pivots[node][1]), pivots[node][2]] \
                    rbuffer[3 - idx][1:length(pivots[node][1]), pivots[child][2]]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(translationidcs, node)
                push!(
                    translations,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], Int.(childs)
                    ),
                )
            end
        end
    end
end

function build_trialbases!(
    translations::Vector{H2BasisBlock{I,K}},
    translationidcs::Vector{Int},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    rbuffer::Matrix{K},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    level,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(level) do node
        if !ClusterTrees.haschildren(tree, node) && pivots[node] != ([], [])
            V =
                rbuffer[1:length(pivots[node][1]), pivots[node][2]] \
                rbuffer[1:length(pivots[node][1]), value(tree, node)]

            lock(lk) do
                push!(momentidcs, node)
                push!(moments, H2BasisBlock(V, pivots[node][1], value(tree, node), Int[]))
            end
        elseif pivots[node] != ([], [])
            translationblocks = Matrix{K}[]
            childs = collect(children(tree, node))
            for child in childs
                Θ =
                    rbuffer[1:length(pivots[node][1]), pivots[node][2]] \
                    rbuffer[1:length(pivots[node][1]), pivots[child][2]]
                push!(translationblocks, Θ)
            end
            lock(lk) do
                push!(translationidcs, node)
                push!(
                    translations,
                    H2BasisBlock(
                        translationblocks, pivots[node][1], pivots[node][2], Int.(childs)
                    ),
                )
            end
        end
    end
end

# zhaocompressor
function build_testbases!(
    leveledtranslations::Vector{Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    cbuffer::Vector{Matrix{K}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (levelidx, level) in enumerate(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        _foreach(level) do node
            if !ClusterTrees.haschildren(tree, node)
                if pivots[node] != ([], [])
                    U =
                        cbuffer[levelidx][value(tree, node), 1:length(pivots[node][2])] /
                        cbuffer[levelidx][pivots[node][1], 1:length(pivots[node][2])]
                    lock(lk) do
                        push!(momentidcs, node)
                        push!(
                            moments,
                            H2BasisBlock(U, value(tree, node), pivots[node][2], Int[]),
                        )
                    end
                end
            elseif pivots[node] != ([], [])
                translationblocks = Matrix{K}[]
                childs = collect(children(tree, node))
                for child in childs
                    for r in pivots[child][1]
                        if 0.0 in cbuffer[levelidx][r, 1:length(pivots[node][2])]
                            println("fail")
                            println(levelidx, ", ", child, ", ", node)
                        end
                    end
                    Θ =
                        cbuffer[levelidx][pivots[child][1], 1:length(pivots[node][2])] /
                        cbuffer[levelidx][pivots[node][1], 1:length(pivots[node][2])]
                    push!(translationblocks, Θ)
                end
                lock(lk) do
                    push!(translationidcs, node)
                    push!(
                        translations,
                        H2BasisBlock(
                            translationblocks,
                            pivots[node][1],
                            pivots[node][2],
                            Int.(childs),
                        ),
                    )
                end
            end
        end
        push!(leveledtranslations, Dict(translationidcs .=> translations))
    end
end

function build_trialbases!(
    leveledtranslations::Vector{Dict{Int,NestedCrossApproximation.H2BasisBlock{Int,K}}},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    rbuffer::Vector{Matrix{K}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    tree;
    multithreading=true,
) where {I,K}
    lk = Threads.SpinLock()

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (levelidx, level) in enumerate(clusterlink)
        translationidcs = Int[]
        translations = NestedCrossApproximation.H2BasisBlock{Int,K}[]
        _foreach(level) do node
            if !ClusterTrees.haschildren(tree, node)
                if pivots[node] != ([], [])
                    V =
                        rbuffer[levelidx][1:length(pivots[node][1]), pivots[node][2]] \
                        rbuffer[levelidx][1:length(pivots[node][1]), value(tree, node)]

                    lock(lk) do
                        push!(momentidcs, node)
                        push!(
                            moments,
                            H2BasisBlock(V, pivots[node][1], value(tree, node), Int[]),
                        )
                    end
                end
            elseif pivots[node] != ([], [])
                translationblocks = Matrix{K}[]
                childs = collect(children(tree, node))
                for child in childs
                    Θ =
                        rbuffer[levelidx][1:length(pivots[node][1]), pivots[node][2]] \
                        rbuffer[levelidx][1:length(pivots[node][1]), pivots[child][2]]
                    push!(translationblocks, Θ)
                end
                lock(lk) do
                    push!(translationidcs, node)
                    push!(
                        translations,
                        H2BasisBlock(
                            translationblocks,
                            pivots[node][1],
                            pivots[node][2],
                            Int.(childs),
                        ),
                    )
                end
            end
        end
        push!(leveledtranslations, Dict(translationidcs .=> translations))
    end
end
