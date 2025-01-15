using OhMyThreads: @spawn, index_chunks
using Base.Threads
using FLoops
using LinearAlgebra

function testfars(
    nnodes::Int, levelfars::Vector{Vector{Tuple{Int,Int}}}
)
    sortedfars = [Int[] for i in 1:nnodes]

    for fars in levelfars
        for far in fars
            push!(sortedfars[far[1]], far[2])
        end
    end

    return sortedfars
end

function fillthisshit_spawn(assembler, clusterlink, tree, sortedfars; nchunks = Threads.nthreads())
    allblocks = Vector{Float64}[]
    C = Channel{Matrix{Float64}}(nthreads())
    for i = 1:nthreads()
        put!(C, zeros(Float64, 200, 3000))
    end
    for level in clusterlink
        tasks = map(index_chunks(1:length(level); n = nchunks)) do idcs 
            @spawn begin
                blocks = Float64[]
                for idx in idcs
                    c = value(tree, sortedfars[level[idx]])
                    if c != []
                        r = value(tree, level[idx])
                        M = take!(C)#zeros(Float64, length(r), length(c))
                        assembler(M, r, c)
                        @inbounds _, s, _ = svd(M[1:length(r), 1:length(c)])
                        push!(blocks, s[1])
                        M[1:length(r), 1:length(c)] .= 0.0
                        put!(C, M)
                    end
                end
                return blocks
            end
        end
        b = fetch.(tasks)
        if b isa Vector{Vector{Float64}}
            append!(allblocks, b)
        end
    end
    return allblocks   
end

function fillthisshit_undef(assembler, clusterlink, tree, sortedfars)
    allblocks = Vector{Float64}(undef, length(tree.nodes))
    for level in clusterlink
       @floop for idx in level
            c = value(tree, sortedfars[idx])
            if c != []
                r = value(tree, idx)
                #println(length(r), " ", length(c))
                M = zeros(Float64, length(r), length(c))
                assembler(M, r, c)
                _, s, _ = svd(M)
                allblocks[idx] = s[1]
            end
        end
    end
    return allblocks   
end

function fillthisshit_undef_channel(assembler, clusterlink, tree, sortedfars)
    C = Channel{Matrix{Float64}}(nthreads())
    for i = 1:nthreads()
        put!(C, zeros(Float64, 200, 3000))
    end
    allblocks = Vector{Float64}(undef, length(tree.nodes))
    for level in clusterlink
       @floop for idx in level
            c = value(tree, sortedfars[idx])
            if c != []
                r = value(tree, idx)
                M = take!(C)
                assembler(M, r, c)
                _, s, _ = svd(M[1:length(r), 1:length(c)])
                allblocks[idx] = s[1]
                M[1:length(r), 1:length(c)] .= 0.0
                put!(C, M)
            end
        end
    end
    return allblocks   
end

##

using BEAST
using FastBEAST
using ClusterTrees
using CompScienceMeshes

Γ = meshsphere(1.0, 0.08)
op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)

tree = create_tree(space.pos, KMeansTreeOptions(nmin=50))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree, η=2.0)

@views farblkassembler = BEAST.blockassembler(
    op, space, space
)
@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkassembler(tdata,sdata,store)
end
##
collect(children(tree, 4))

##
cl = FastBEAST.cluster_link(tree)
sfars = testfars(length(tree.nodes), fars)

@time X = fillthisshit_spawn(farassembler, cl[1:6], tree, sfars);
@time X2 = fillthisshit_undef(farassembler, cl[1:6], tree, sfars);
@time X3 = fillthisshit_undef_channel(farassembler, cl[1:6], tree, sfars);

