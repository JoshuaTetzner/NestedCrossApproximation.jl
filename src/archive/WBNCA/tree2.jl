using FastBEAST
using ClusterTrees
import FastBEAST.NminClusterTrees.NminTree

function islf(
    tnode::ClusterTrees.LevelledTrees.HNode{D},
    snode::ClusterTrees.LevelledTrees.HNode{D},
    k::K,
) where {D,K}
    ths = tnode.node.data.hs
    shs = snode.node.data.hs

    k / pi * 4 * min(ths, shs) <= 1 ? (return true) : (return false)
end

function islf(tnode::ClusterTrees.LevelledTrees.HNode{D}, k::K) where {D,K}
    ths = tnode.node.data.hs

    k / pi * 4 * hs <= 1 ? (return true) : (return false)
end

function islffar(
    tnode::ClusterTrees.LevelledTrees.HNode{D},
    snode::ClusterTrees.LevelledTrees.HNode{D};
    η=1.0,
) where {D}
    ths = tnode.node.data.hs
    shs = snode.node.data.hs
    dist = norm(tnode.node.data.ct - snode.node.data.ct) - (ths + shs)

    (2 * max(ths, shs) <= η * max(dist, 0.0)) ? (return true) : (return false)
end

function ishffar(
    tnode::ClusterTrees.LevelledTrees.HNode{D},
    snode::ClusterTrees.LevelledTrees.HNode{D},
    k::K;
    η=1.0,
) where {D,K}
    ths = tnode.node.data.hs
    shs = snode.node.data.hs
    dist = norm(tnode.node.data.ct - snode.node.data.ct) - (ths + shs)

    (4 * k * max(ths^2, shs^2) <= η * max(dist, 0.0)) ? (return true) : (return false)
end

function listnearfarinteractionshf(
    block_tree::ClusterTrees.BlockTrees.BlockTree{T},
    block,
    state,
    nears::Vector{Tuple{Int,Int}},
    lf_fars::Vector{Vector{Tuple{Int,Int}}},
    hf_fars::Vector{Vector{Tuple{Int,Int}}},
    level::Int,
    k::K;
    ηₕ=1.0,
    ηₗ=1.0,
) where {T,K}
    if islf(state[1], state[2], k)
        islffar(state[1], state[2]; η=ηₗ) && (push!(lf_fars[level], block); return nothing)
        !ClusterTrees.haschildren(block_tree, block) &&
            (push!(nears, block); return nothing)
        for chd in ClusterTrees.children(block_tree, block)
            chd_state = (
                block_tree.test_cluster.nodes[chd[1]],
                block_tree.trial_cluster.nodes[chd[2]],
            )
            listnearfarinteractionshf(
                block_tree,
                chd,
                chd_state,
                nears,
                lf_fars,
                hf_fars,
                level + 1,
                k;
                ηₕ=ηₕ,
                ηₗ=ηₗ,
            )
        end
    else
        ishffar(state[1], state[2], k; η=ηₕ) &&
            (push!(hf_fars[level], block); return nothing)
        !ClusterTrees.haschildren(block_tree, block) &&
            (push!(nears, block); return nothing)
        for chd in ClusterTrees.children(block_tree, block)
            chd_state = (
                block_tree.test_cluster.nodes[chd[1]],
                block_tree.trial_cluster.nodes[chd[2]],
            )
            listnearfarinteractionshf(
                block_tree,
                chd,
                chd_state,
                nears,
                lf_fars,
                hf_fars,
                level + 1,
                k;
                ηₕ=ηₕ,
                ηₗ=ηₗ,
            )
        end
    end
end

function computeinteractionshf(
    tree::ClusterTrees.BlockTrees.BlockTree{T}, k::K; ηₕ=1.0, ηₗ=1.0
) where {T,K}
    nears = Tuple{Int,Int}[]
    num_levels = length(tree.test_cluster.levels)
    lf_fars = [Tuple{Int,Int}[] for l in 1:num_levels]
    hf_fars = [Tuple{Int,Int}[] for l in 1:num_levels]

    root_state = (tree.test_cluster.nodes[1], tree.trial_cluster.nodes[1])
    root_level = 1

    listnearfarinteractionshf(
        tree,
        ClusterTrees.root(tree),
        root_state,
        nears,
        lf_fars,
        hf_fars,
        root_level,
        k;
        ηₕ=ηₕ,
        ηₗ=ηₗ,
    )
    return nears, hf_fars, lf_fars
end

function sortinteractions(fars::Vector{Vector{Tuple{Int,Int}}})
    levelidcs = Int[]
    tdicts = Dict{Int,Vector{Int}}[]
    sdicts = Dict{Int,Vector{Int}}[]
    for (levelidx, level) in enumerate(fars)
        level == [] && continue
        push!(levelidcs, levelidx)
        sort!(level)
        t = [level[1][1]]
        Ft = [Int[]]
        for interaction in level
            if t[end] == interaction[1]
                push!(Ft[end], interaction[2])
            else
                push!(t, interaction[1])
                push!(Ft, [interaction[2]])
            end
        end
        sort!(level; by=x -> x[2])
        s = [level[1][2]]
        Fs = [Int[]]
        for interaction in level
            if s[end] == interaction[2]
                push!(Fs[end], interaction[1])
            else
                push!(s, interaction[2])
                push!(Fs, [interaction[1]])
            end
        end
        push!(tdicts, Dict(t .=> Ft))
        push!(sdicts, Dict(s .=> Fs))
    end
    return Dict(levelidcs .=> tdicts), Dict(levelidcs .=> sdicts)
end

function sortinteractionsWNCA(
    tree::ClusterTrees.BlockTrees.BlockTree{T}, fars::Vector{Vector{Tuple{Int,Int}}}
) where {T}
    tfars = [Int[] for _ in 1:length(tree.test_cluster.nodes)]
    sfars = [Int[] for _ in 1:length(tree.trial_cluster.nodes)]
    for level in fars
        level == [] && continue
        for interaction in level
            push!(tfars[interaction[1]], interaction[2])
            push!(sfars[interaction[2]], interaction[1])
        end
    end

    #=testcl = FastBEAST.cluster_link(tree.test_cluster)
    trialcl = FastBEAST.cluster_link(tree.test_cluster)
    for level in testcl[2:end]
        for cluster in level
            append!(tfars[cluster], tfars[ClusterTrees.parent(tree.test_cluster, cluster)])
        end
    end
    for level in trialcl[2:end]
        for cluster in level
            append!(sfars[cluster], sfars[ClusterTrees.parent(tree.trial_cluster, cluster)])
        end
    end=#
    return tfars, sfars
end
