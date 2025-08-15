# Number of directions
radius = 1.0
Γ = meshicosphere(200, radius)
Γ.vertices
tree = create_tree(Γ.vertices, BoxTreeOptions(; nmin=50))
λ = 0.5
k = 2 * pi / λ
η₁ = 1

δₗ(l) = sqrt(3) * 2 / 2^l
dsquare(l) = η₁ / (k * δₗ(l))

# Level 0
dsquare(2)
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
tree.levels

nears, hffars, lffars = computeinteractionshf(blktree, k; ηₕ=5.0)

hffars
sort(lffars[end]; by=x -> x[2])

sort(lffars)

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

@time tfars, sfars = sortinteractions(hffars);
@time tfars, sfars = sortinteractions(length(tree.nodes), length(tree.nodes), hffars);
@time lffars, sffars = sortinteractions(hffars);

##
@time findfirst(!=(Int[]), hffars)
findlast(!=(Int[]), hffars)

##
radius = 1.0
Γ = meshicosphere(200, radius)
Γ.vertices
tree = create_tree(Γ.vertices, BoxTreeOptions(; nmin=50))
λ = 0.1
k = 2 * pi / λ
η₁ = 1

δₗ(l) = sqrt(3) * 2 / 2^l
dsquare(l) = η₁ / (k * δₗ(l))

tree = directionaltree(2.0, k, 5.0)

tree.nodes[7].data.ℯ
for child in ClusterTrees.children(tree, 7)
    println(tree.nodes[child].data.ℯ)
end

##
Γ = meshicosphere(200, radius)
Γ.vertices
tree = create_tree(Γ.vertices, BoxTreeOptions(; nmin=50))
λ = 0.5
k = 2 * pi / λ
η₁ = 1
