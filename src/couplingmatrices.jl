function assemble_couplingstore(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::AbstractVector{<:AbstractVector{Int}},
    trialpivots::AbstractVector{<:AbstractVector{Int}},
    testfardata::FarData,
    trialfardata::FarData;
    scheduler=DynamicScheduler(),
    pbar=Progress(0; enabled=false),
) where {T}
    ptr = farptr(testfardata)
    nodes = farnodes(testfardata)
    nfars = length(nodes)
    isempty(nodes) &&
        return CouplingStore{T}(CouplingTraversalPlan(Int.(ptr), Int.(nodes)), Matrix{T}[])

    nnodes = length(ptr) - 1
    couplingmatrices = Vector{Matrix{T}}(undef, nfars)
    @inbounds for tnode in 1:nnodes
        for i in ptr[tnode]:(ptr[tnode + 1] - 1)
            snode = nodes[i]
            couplingmatrices[i] = zeros(
                T, length(testpivots[tnode]), length(trialpivots[snode])
            )
        end
    end

    @tasks for tnode in 1:nnodes
        @set scheduler = scheduler
        for i in ptr[tnode]:(ptr[tnode + 1] - 1)
            snode = nodes[i]
            if !isempty(testpivots[tnode]) && !isempty(trialpivots[snode])
                farmatrix(couplingmatrices[i], testpivots[tnode], trialpivots[snode])
            end
            next!(pbar)
        end
    end

    return CouplingStore{T}(CouplingTraversalPlan(Int.(ptr), Int.(nodes)), couplingmatrices)
end

function assemble_couplingstore(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::AbstractVector{<:AbstractVector{Int}},
    trialpivots::AbstractVector{<:AbstractVector{Int}},
    testfardata::DirectionalData,
    trialfardata::DirectionalData;
    scheduler=DynamicScheduler(),
    pbar=Progress(0; enabled=false),
) where {T}
    testptr = farptr(testfardata)
    trialptr = farptr(trialfardata)
    testnodes = farnodes(testfardata)
    trialnodes = farnodes(trialfardata)
    nfars = length(testnodes)
    nnodes = length(testptr) - 1

    couplingmatrices = Vector{Matrix{T}}(undef, nfars)
    sdiridcs = Vector{Int}(undef, nfars)
    tdiridcs = Vector{Int}(undef, nfars)
    @tasks for t in 1:nnodes
        @set scheduler = scheduler
        for (localsidx, sidx) in enumerate(testptr[t]:(testptr[t + 1] - 1))
            s = testnodes[sidx]
            tdiridx = diridxfromlocalfaridx(testfardata, t, localsidx)
            localtidx = findfirst(
                ==(t), view(trialnodes, trialptr[s]:(trialptr[s + 1] - 1))
            )
            sdiridx = diridxfromlocalfaridx(trialfardata, s, localtidx)
            couplingmatrices[sidx] = zeros(
                T, length(testpivots[tdiridx]), length(trialpivots[sdiridx])
            )

            sdiridcs[sidx] = sdiridx
            tdiridcs[sidx] = tdiridx

            if !isempty(testpivots[tdiridx]) && !isempty(trialpivots[sdiridx])
                farmatrix(couplingmatrices[sidx], testpivots[tdiridx], trialpivots[sdiridx])
            end
            next!(pbar)
        end
    end

    return CouplingStore{T}(
        DirCouplingTraversalPlan(Int.(testptr), Int.(tdiridcs), Int.(sdiridcs)),
        couplingmatrices,
    )
end
