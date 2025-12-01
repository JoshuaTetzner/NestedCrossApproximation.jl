function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    tree::H2Trees.BlockTree,
    testpivots::Vector{Tuple{Vector{Int},Vector{Int}}},
    trialpivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    isnear=H2Trees.isnear,
    ntasks=1,
) where {T}
    lk = Threads.SpinLock()
    lowrankblocks = H2MatrixBlock{Int,T}[]
    fars = Vector{Tuple{Int,Int}}[]
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)

    for level in H2Trees.levels(H2Trees.testtree(tree))
        levelfars = Tuple{Int,Int}[]
        testclusters = collect(H2Trees.LevelIterator(H2Trees.testtree(tree), level))
        @tasks for t in testclusters
            @set ntasks = ntasks
            for s in iterator(H2Trees.testtree(tree), H2Trees.trialtree(tree), t)
                blk = zeros(T, length(testpivots[t][1]), length(trialpivots[s][2]))
                farmatrix(blk, testpivots[t][1], trialpivots[s][2])
                lock(lk) do
                    push!(levelfars, (t, s))
                    push!(lowrankblocks, H2MatrixBlock(blk, t, s))
                end
            end
        end
        push!(fars, levelfars)
    end

    return lowrankblocks, fars
end

struct DH2MatrixBlock{I,K}
    Z::Matrix{K}
    row_basis::I
    col_basis::I
    dir::Int
end

function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    testpivots,
    trialpivots,
    fars::Vector{Vector{Int}},
    e::Vector{Vector{Int}};
    ntasks=Threads.nthreads(),
) where {T}
    lk = Threads.SpinLock()
    couplingblocks = DH2MatrixBlock{Int,T}[]

    for (t, Ft) in enumerate(fars)
        @tasks for sidx in eachindex(Ft)
            @set ntasks = ntasks
            blk = zeros(
                T,
                length(testpivots[t][e[t][sidx]][1]),
                length(trialpivots[Ft[sidx]][e[t][sidx]][2]),
            )
            farmatrix(
                blk, testpivots[t][e[t][sidx]][1], trialpivots[Ft[sidx]][e[t][sidx]][2]
            )
            lock(lk) do
                push!(couplingblocks, DH2MatrixBlock(blk, t, Ft[sidx], e[t][sidx]))
            end
        end
    end

    return couplingblocks
end

#=
function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    testpivots,
    trialpivots,
    tree::BlockTree,
    dtree::𝒟tree;
    #fars::Vector{Vector{Int}},
    #e::Vector{Vector{Int}};
    isnear=H2Trees.isnear,
    ntasks=Threads.nthreads(),
) where {T}
    lk = Threads.SpinLock()
    couplingblocks = DH2MatrixBlock{Int,T}[]
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)

    for level in levels(testtree(tree))
        @tasks for t in collect(LevelIterator(testtree(tree), level))
            @set ntasks = ntasks
            for s in iterator(trialtree(tree), testtree(tree), t)
                e = direction(
                    center(trialtree(tree), s) - center(testtree(tree), t),
                    dtree,
                    max(0, dtree.level + 1 - level),
                )
                blk = zeros(T, length(testpivots[t][e][1]), length(trialpivots[s][e][2]))
                farmatrix(blk, testpivots[t][e][1], trialpivots[s][e][2])
                lock(lk) do
                    push!(couplingblocks, DH2MatrixBlock(blk, t, s, e))
                end
            end
        end
    end

    return couplingblocks
end
=#
