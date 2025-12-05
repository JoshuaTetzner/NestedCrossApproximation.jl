function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    tree,
    testpivots::Vector{Tuple{Vector{Int},Vector{Int}}},
    trialpivots::Vector{Tuple{Vector{Int},Vector{Int}}};
    isnear=H2Trees.isnear,
    ntasks=Threads.nthreads(),
) where {T}
    couplingmatrices = Vector{Vector{Tuple{Int,Matrix{T}}}}(undef, length(testpivots))
    iterator = H2Trees.WellSeparatedIterator(; isnear=(tree) -> isnear)(tree)

    for level in H2Trees.levels(testtree(tree))
        @tasks for t in tcollect(LevelIterator(testtree(tree), level))
            @set ntasks = ntasks
            ncmats = Tuple{Int,Matrix{T}}[]
            for s in iterator(H2Trees.trialtree(tree), H2Trees.testtree(tree), t)
                blk = zeros(T, length(testpivots[t][1]), length(trialpivots[s][2]))
                farmatrix(blk, testpivots[t][1], trialpivots[s][2])
                push!(ncmats, (s, blk))
            end
            ncmats != [] && (couplingmatrices[t] = ncmats)
        end
    end

    return couplingmatrices
end

struct DH2MatrixBlock{I,K}
    Z::Matrix{K}
    row_basis::I
    col_basis::I
    dir::Tuple{Int,Int}
end

function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::Vector{D},
    trialpivots::Vector{D},
    testdata::DirectionalData,
    trialdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {T,D<:Dict{Int,Tuple{Vector{Int},Vector{Int}}}}
    cmats = Vector{Dict{Tuple{Int,Int},Matrix{T}}}(undef, length(testpivots))

    @tasks for t in eachindex(testdata.F)
        @set ntasks = ntasks
        cdirs = Tuple{Int,Int}[]
        ncmats = Matrix{T}[]
        for sidx in eachindex(testdata.F[t])
            s = testdata.F[t][sidx]
            eₜ = testdata.𝓔[t][sidx]
            eₛ = trialdata.𝓔[s][findfirst(x -> x == t, trialdata.F[s])]
            blk = zeros(T, length(testpivots[t][eₜ][1]), length(trialpivots[s][eₛ][2]))
            farmatrix(blk, testpivots[t][eₜ][1], trialpivots[s][eₛ][2])
            push!(ncmats, blk)
            push!(cdirs, (eₜ, eₛ))
        end
        cmats[t] = Dict(cdirs .=> ncmats)
    end

    return cmats
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
