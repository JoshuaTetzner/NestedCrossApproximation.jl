function build_testh2blocks!(
    translations::Vector{H2BasisBlock{I,K}},
    translationidcs::Vector{Int},
    moments::Vector{H2BasisBlock{I,K}},
    momentidcs::Vector{Int},
    cbuffer::Tuple{Matrix{Float64},Matrix{Float64}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    fars,
    level,
    tree;
    multithreading=true,
) where {I,K}
    lk = ReentrantLock()
    iseven(level) ? idx = 1 : idx = 2

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(clusterlink[level]) do node
        if !ClusterTrees.haschildren(tree, node)
            if pivots[node] != ([], [])
                V =
                    cbuffer[idx][value(tree, node), 1:length(pivots[node][1])] *
                    inv(cbuffer[idx][pivots[node][1], 1:length(pivots[node][2])])
                lock(lk) do
                    push!(momentidcs, node)
                    push!(
                        moments, H2BasisBlock(V, value(tree, node), pivots[node][2], Int[])
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
                    cbuffer[3 - idx][pivots[child][1], 1:length(pivots[node][2])] *
                    inv(cbuffer[3 - idx][pivots[node][1], 1:length(pivots[node][2])])
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
    end
end

function build_i2itranslations!(
    i2itranslations::Vector{H2BasisBlock{I,K}},
    rbuffer::Tuple{Matrix{K},Matrix{K}},
    pivots::Vector{Tuple{Vector{I},Vector{I}}},
    clusterlink,
    fars,
    level,
    tree;
    multithreading=true,
) where {I,K}
    iseven(level) ? idx = 1 : idx = 2
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    _foreach(clusterlink[level]) do node
        if fars[node] != []
            childs = collect(children(tree, node))
            translations = Matrix{K}[]
            for child in childs
                @inbounds Θ =
                    inv(rbuffer[idx][1:length(pivots[node][1]), pivots[node][2]]) *
                    rbuffer[idx][1:length(pivots[node][1]), pivots[child][2]]
                push!(translations, Θ)
            end
            i2itranslations[node] = H2BasisBlock(
                translations, pivots[node][1], pivots[node][2], childs
            )
        end
    end
end

function I2Otranslations(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{Int,Int}}},
    pivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    multithreading=true,
) where {K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{Int,K}}(undef, length(fars))

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        blk = zeros(K, length(pivots[far[1]][1]), length(pivots[far[2]][1]))
        matrixassembler(blk, pivots[far[1]][1], pivots[far[2]][1])
        lowrankblocks[idx] = H2MatrixBlock(blk, far[1], far[2])
    end

    return lowrankblocks
end
