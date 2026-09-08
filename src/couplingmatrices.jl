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
    rowidcs = Vector{Vector{Int}}(undef, nfars)
    colidcs = Vector{Vector{Int}}(undef, nfars)
    @inbounds for tnode in 1:nnodes
        for i in ptr[tnode]:(ptr[tnode + 1] - 1)
            snode = nodes[i]
            rowidcs[i] = testpivots[tnode]
            colidcs[i] = trialpivots[snode]
            couplingmatrices[i] = zeros(T, length(rowidcs[i]), length(colidcs[i]))
        end
    end

    # Nodes without a selected far-field rank must be skipped: farmatrix(_, [], _) is
    # not a safe no-op for every backend.
    active = filter(i -> !isempty(rowidcs[i]) && !isempty(colidcs[i]), 1:nfars)
    assemble_blocks(
        farmatrix,
        view(couplingmatrices, active),
        view(rowidcs, active),
        view(colidcs, active);
        scheduler=scheduler,
        pbar=pbar,
    )
    next!(pbar; step=nfars - length(active))

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
    rowidcs = Vector{Vector{Int}}(undef, nfars)
    colidcs = Vector{Vector{Int}}(undef, nfars)
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

            rowidcs[sidx] = testpivots[tdiridx]
            colidcs[sidx] = trialpivots[sdiridx]
            couplingmatrices[sidx] = zeros(T, length(rowidcs[sidx]), length(colidcs[sidx]))
            sdiridcs[sidx] = sdiridx
            tdiridcs[sidx] = tdiridx
        end
    end

    # Empty pivot sets must be skipped, see the FarData method above.
    active = filter(i -> !isempty(rowidcs[i]) && !isempty(colidcs[i]), 1:nfars)
    assemble_blocks(
        farmatrix,
        view(couplingmatrices, active),
        view(rowidcs, active),
        view(colidcs, active);
        scheduler=scheduler,
        pbar=pbar,
    )
    next!(pbar; step=nfars - length(active))

    return CouplingStore{T}(
        DirCouplingTraversalPlan(Int.(testptr), Int.(tdiridcs), Int.(sdiridcs)),
        couplingmatrices,
    )
end
