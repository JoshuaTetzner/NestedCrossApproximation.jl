function assemble_couplingstore(
    farmatrix::AbstractKernelMatrix{T},
    testpivots::AbstractVector{<:AbstractVector{Int}},
    trialpivots::AbstractVector{<:AbstractVector{Int}},
    testfardata::FarData;
    scheduler=DynamicScheduler(),
) where {T}
    fp = farptr(testfardata)
    fs = fars(testfardata)
    nfars = length(fs)
    isempty(fs) && return CouplingStore{T}(CouplingTraversalPlan(Int.(fp), Int.(fs)), Matrix{T}[])

    nnodes = length(fp) - 1
    couplingmatrices = Vector{Matrix{T}}(undef, nfars)
    @inbounds for tnode in 1:nnodes
        for i in fp[tnode]:(fp[tnode + 1] - 1)
            snode = fs[i]
            couplingmatrices[i] = zeros(T, length(testpivots[tnode]), length(trialpivots[snode]))
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
