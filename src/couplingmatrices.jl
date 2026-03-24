function assemble_couplingstore(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::AbstractVector{<:AbstractVector{Int}},
    trialpivots::AbstractVector{<:AbstractVector{Int}},
    testfardata::FarData,
    trialfardata::FarData;
    scheduler=DynamicScheduler(),
) where {T}
    fp = farptr(testfardata)
    fs = fars(testfardata)
    nfars = length(fs)
    isempty(fs) &&
        return CouplingStore{T}(CouplingTraversalPlan(Int.(fp), Int.(fs)), Matrix{T}[])

    nnodes = length(fp) - 1
    couplingmatrices = Vector{Matrix{T}}(undef, nfars)
    @inbounds for tnode in 1:nnodes
        for i in fp[tnode]:(fp[tnode + 1] - 1)
            snode = fs[i]
            couplingmatrices[i] = zeros(
                T, length(testpivots[tnode]), length(trialpivots[snode])
            )
        end
    end

    @tasks for tnode in 1:nnodes
        @set scheduler = scheduler
        for i in fp[tnode]:(fp[tnode + 1] - 1)
            snode = fs[i]
            if !isempty(testpivots[tnode]) && !isempty(trialpivots[snode])
                farmatrix(couplingmatrices[i], testpivots[tnode], trialpivots[snode])
            end
        end
    end

    return CouplingStore{T}(CouplingTraversalPlan(Int.(fp), Int.(fs)), couplingmatrices)
end

function assemble_couplingstore(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::AbstractVector{<:AbstractVector{Int}},
    trialpivots::AbstractVector{<:AbstractVector{Int}},
    testfardata::DirectionalData,
    trialfardata::DirectionalData;
    scheduler=DynamicScheduler(),
) where {T}
    testfp = farptr(testfardata)
    trialfp = farptr(trialfardata)
    testfs = fars(testfardata)
    trialfs = fars(trialfardata)
    nfars = length(testfs)
    nnodes = length(testfp) - 1

    couplingmatrices = Vector{Matrix{T}}(undef, nfars)
    sdiridcs = Vector{Int}(undef, nfars)
    tdiridcs = Vector{Int}(undef, nfars)
    @tasks for t in 1:nnodes
        @set scheduler = scheduler
        for (localsidx, sidx) in enumerate(testfp[t]:(testfp[t + 1] - 1))
            s = testfs[sidx]
            tdiridx = diridxfromlocalfaridx(testfardata, t, localsidx)
            localtidx = findfirst(==(t), view(trialfs, trialfp[s]:(trialfp[s + 1] - 1)))
            sdiridx = diridxfromlocalfaridx(trialfardata, s, localtidx)
            couplingmatrices[sidx] = zeros(
                T, length(testpivots[tdiridx]), length(trialpivots[sdiridx])
            )

            sdiridcs[sidx] = sdiridx
            tdiridcs[sidx] = tdiridx

            farmatrix(couplingmatrices[sidx], testpivots[tdiridx], trialpivots[sdiridx])
        end
    end

    return CouplingStore{T}(
        DirCouplingTraversalPlan(Int.(testfp), Int.(tdiridcs), Int.(sdiridcs)),
        couplingmatrices,
    )
end
