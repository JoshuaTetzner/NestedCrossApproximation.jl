function _build_transfer_store(
    transfer::Vector{Vector{Matrix{T}}},
    level_transfer_nodes::Vector{Vector{Int}},
    level_child_counts::Vector{Vector{Int}},
    tree,
) where {T}
    nlevels = length(level_transfer_nodes)
    level_ptr = Vector{Int}(undef, nlevels + 1)
    level_ptr[1] = 1
    for level in 1:nlevels
        level_ptr[level + 1] = level_ptr[level] + length(level_transfer_nodes[level])
    end

    nnodes = level_ptr[end] - 1
    level_nodes = Vector{Int}(undef, nnodes)
    node_child_counts = Vector{Int}(undef, nnodes)
    idx = 1
    @inbounds for level in 1:nlevels
        nodes = level_transfer_nodes[level]
        counts = level_child_counts[level]
        for i in eachindex(nodes)
            level_nodes[idx] = nodes[i]
            node_child_counts[idx] = counts[i]
            idx += 1
        end
    end

    node_ptr = Vector{Int}(undef, nnodes + 1)
    node_ptr[1] = 1
    for i in 1:nnodes
        node_ptr[i + 1] = node_ptr[i] + node_child_counts[i]
    end

    nedges = node_ptr[end] - 1
    edge_child = Vector{Int}(undef, nedges)
    blocks = Vector{Matrix{T}}(undef, nedges)

    e = 1
    @inbounds for i in 1:nnodes
        node = level_nodes[i]
        children = collect(H2Trees.ChildIterator(tree, node))
        tblocks = transfer[node]
        for j in eachindex(children)
            edge_child[e] = children[j]
            blocks[e] = tblocks[j]
            e += 1
        end
    end

    plan = TransferTraversalPlan(level_ptr, level_nodes, node_ptr, edge_child)
    return TransferStore{T}(plan, blocks)
end

function testtransfermatrices!(
    tree, transfer, levelnodes, pivots, buf; scheduler=DynamicScheduler()
)
    levelnodecounts = Vector{Int}(undef, length(levelnodes))
    @tasks for nodeidx in eachindex(levelnodes)
        @set scheduler = scheduler
        node = levelnodes[nodeidx]
        children = collect(H2Trees.ChildIterator(tree, node))
        nodetransfers = Vector{Matrix{eltype(buf)}}(undef, length(children))
        @inbounds for j in eachindex(children)
            child = children[j]
            nodetransfers[j] = testtransfer(pivots[node], pivots[child], buf)
        end
        transfer[node] = nodetransfers
        levelnodecounts[nodeidx] = length(children)
    end
    return levelnodecounts
end

function testtransfer(rows, childrows, buf)
    return buf[childrows, 1:length(rows)] / buf[rows, 1:length(rows)]
end

function trialtransfermatrices!(
    tree, transfer, levelnodes, pivots, buf; scheduler=DynamicScheduler()
)
    levelnodecounts = Vector{Int}(undef, length(levelnodes))
    @tasks for nodeidx in eachindex(levelnodes)
        @set scheduler = scheduler
        node = levelnodes[nodeidx]
        children = collect(H2Trees.ChildIterator(tree, node))
        nodetransfers = Vector{Matrix{eltype(buf)}}(undef, length(children))
        @inbounds for j in eachindex(children)
            child = children[j]
            nodetransfers[j] = trialtransfer(pivots[node], pivots[child], buf)
        end
        transfer[node] = nodetransfers
        levelnodecounts[nodeidx] = length(children)
    end
    return levelnodecounts
end

function trialtransfer(cols, childcols, buf)
    return buf[1:length(cols), cols] \ buf[1:length(cols), childcols]
end

#=
# directional topdowncompressor
function testtransfermatrix(
    t::Int,
    dir::Int,
    pivs::Tuple{Vector{Int},Vector{Int}},
    buffer::Matrix{K},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData,
) where {I,K}
    tmats = Tuple{Int,Matrix{K}}[]
    for child in ChildIterator(tree, t)
        crows = pivots[child][paternaldirection(dirdata, child, dir)][1]
        rows = pivs[1]

        @assert length(crows) == length(unique(crows))
        @assert length(rows) == length(unique(rows))

        @views tmat = buffer[crows, 1:length(rows)] / buffer[rows, 1:length(rows)]
        push!(tmats, (paternaldirection(dirdata, child, dir), tmat))
    end
    return tmats
end

function testtransfermatrices!(
    tmats::Vector{Dict{Int,Vector{Tuple{Int,Matrix{K}}}}},
    level::Int,
    blocks::Vector{D},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {I,K,D<:Dict{Int,Matrix{K}}}
    @tasks for t in collect(H2Trees.LevelIterator(tree, level))
        @set ntasks = ntasks
        if !isdirectionalroot(dirdata, tree, t) && isassigned(pivots, t)
            ntmats = Vector{Tuple{Int,Matrix{K}}}[]
            for (dir, piv) in pivots[t]
                dntmats = Tuple{Int,Matrix{K}}[]
                #cdirs = Int[]
                for child in ChildIterator(tree, t)
                    crows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for
                        idx in pivots[child][paternaldirection(dirdata, child, dir)][1]
                    ]
                    rows = [
                        findfirst(x -> x == idx, H2Trees.values(tree, t)) for idx in piv[1]
                    ]
                    tmat = blocks[t][dir][crows, :] / blocks[t][dir][rows, :]

                    push!(dntmats, (paternaldirection(dirdata, child, dir), tmat))
                end
                push!(ntmats, dntmats)
            end
            tmats[t] = Dict(keys(pivots[t]) .=> ntmats)
        end
    end
end

# directional topdowncompressor
function trialtransfermatrix(
    s::Int,
    dir::Int,
    pivs,
    buffer::Matrix{K},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData,
) where {I,K}
    tmats = Tuple{Int,Matrix{K}}[]
    for child in ChildIterator(tree, s)
        ccols = pivots[child][paternaldirection(dirdata, child, dir)][2]
        cols = pivs[2]
        @views tmat = buffer[1:length(cols), cols] \ buffer[1:length(cols), ccols]

        push!(tmats, (paternaldirection(dirdata, child, dir), tmat))
    end
    return tmats
end

function trialtransfermatrices!(
    tmats::Vector{Dict{Int,Vector{Tuple{Int,Matrix{K}}}}},
    level::Int,
    blocks::Vector{D},
    pivots::Vector{Dict{Int,Tuple{Vector{I},Vector{I}}}},
    tree,
    dirdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {I,K,D<:Dict{Int,Matrix{K}}}
    @tasks for s in collect(H2Trees.LevelIterator(tree, level))
        @set ntasks = ntasks
        if !isdirectionalroot(dirdata, tree, s) && isassigned(pivots, s)
            ntmats = Vector{Tuple{Int,Matrix{K}}}[]
            for (dir, piv) in pivots[s]
                dntmats = Tuple{Int,Matrix{K}}[]
                for child in ChildIterator(tree, s)
                    ccols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for
                        idx in pivots[child][paternaldirection(dirdata, child, dir)][2]
                    ]
                    cols = [
                        findfirst(x -> x == idx, H2Trees.values(tree, s)) for idx in piv[2]
                    ]
                    tmat = blocks[s][dir][:, cols] \ blocks[s][dir][:, ccols]

                    push!(dntmats, (paternaldirection(dirdata, child, dir), tmat))
                end
                push!(ntmats, dntmats)
            end
            tmats[s] = Dict(keys(pivots[s]) .=> ntmats)
        end
    end
end
=#
