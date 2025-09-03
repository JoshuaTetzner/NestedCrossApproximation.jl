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
