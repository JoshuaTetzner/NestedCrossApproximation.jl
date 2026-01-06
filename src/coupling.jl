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
        @tasks for t in collect(LevelIterator(testtree(tree), level))
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

function assemble_couplingmatrices(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::Vector{D},
    trialpivots::Vector{D},
    testdata::DirectionalData,
    trialdata::DirectionalData;
    ntasks=Threads.nthreads(),
) where {T,D<:Dict{Int,Tuple{Vector{Int},Vector{Int}}}}
    cmats = Vector{Dict{Tuple{Int,Int},Pair{Tuple{Int,Int},Matrix{T}}}}(
        undef, length(testpivots)
    )

    @tasks for t in eachindex(testdata.F)
        @set ntasks = ntasks
        cidcs = Tuple{Int,Int}[]
        cdirs = Tuple{Int,Int}[]
        ncmats = Matrix{T}[]
        for (sidx, s) in enumerate(testdata.F[t])
            testdata.𝓔[t] == [0] ? (eₜ = 0) : (eₜ = testdata.𝓔[t][sidx])
            if trialdata.𝓔[s] == [0]
                eₛ = 0
            else
                eₛ = trialdata.𝓔[s][findfirst(x -> x == t, trialdata.F[s])]
            end
            blk = zeros(T, length(testpivots[t][eₜ][1]), length(trialpivots[s][eₛ][2]))
            farmatrix(blk, testpivots[t][eₜ][1], trialpivots[s][eₛ][2])
            push!(ncmats, blk)
            push!(cidcs, (t, s))
            push!(cdirs, (eₜ, eₛ))
        end

        cmats[t] = Dict(cidcs .=> cdirs .=> ncmats)
    end

    return cmats
end
