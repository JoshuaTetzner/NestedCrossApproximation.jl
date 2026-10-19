using H2Trees: isleaf
using ProgressMeter: next!
using NestedCrossApproximation.OhMyThreads: @tasks, @set, DynamicScheduler, SerialScheduler

# ===========================================================================
# Phase 3 (GPU): compress each device's level shard concurrently.
#
# Shared machinery (test and trial differ only in a final transpose):
#   1. `_compress` copies the pristine buffer, runs the batched fully-pivoted
#      ACA (ACACUDA method on the ACA callable `cfg`) on the copy, and brings the
#      small pivot/rank arrays back to host.
#   2. `_solve_transposed_factors` builds, from the *pristine* buffer, each
#      block's L_i = A_i[:,colpivots] (m×r) and its pivot-row square
#      P_i = L_i[rowpivots,:] (r×r), then does one batched LU solve
#          Pᵢᵀ Z = Lᵢᵀ   ⇒   Z = Pᵢ⁻ᵀ Lᵢᵀ = (Lᵢ Pᵢ⁻¹)ᵀ = Xᵢᵀ   (r×m per block)
#      Identity-padded to the largest selected rank in the shard so all blocks
#      factor/solve as one uniform batch with
#      `getrf_strided_batched!` / `getrs_strided_batched!('T', …)`.
#
# The single `Z = Xᵢᵀ` serves both sides (see `src/nestedbasis.jl` /
# `src/transfermatrices.jl`):
#   * test  basis   = Xᵢ            = transpose(Z)            (m×r)
#     test  transfer = Xᵢ[childrows] = transpose(Z[:,childrng]) (childlen×r)
#   * trial basis   = Xᵢᵀ           = Z                        (r×m)
#     trial transfer = Xᵢᵀ[:,child] = Z[:,childrng]            (r×childlen)
# ===========================================================================

# Copy the level buffer (ACA overwrites) and run the batched fully-pivoted ACA on
# the copy; return host `ranks`/`rowpivots` (for pivot bookkeeping) and the
# device pivots (for the gather below), plus the solved transposed factors `Z`.
function _compress(cfg, stack::DistributedBlockStack)
    CUDA.device!(stack.device)
    workingcopy = copy(stack.buffer)
    ranks_d, rowpivots_d, colpivots_d = cfg(
        workingcopy, stack.rowoffsets, stack.ncols, stack.maxrank; sampled=true
    )
    ranks = Int.(Array(ranks_d))
    rowpivots = Int.(Array(rowpivots_d))
    # ACA still samples stack.maxrank columns. Only the solve padding shrinks.
    solverank = max(1, maximum(ranks; init=0))
    Z, _maxM = _solve_transposed_factors(
        stack, ranks_d, rowpivots_d, colpivots_d; maxrank=solverank
    )
    return ranks, rowpivots, Z
end

# Gather the pivot cross matrices and transposed columns in one level-wide
# launch. Each tensor entry is written, including identity/zero padding.
_deviceindices(x::CUDA.AnyCuArray) = x
_deviceindices(x) = CuArray(x)

function _gather_solve_tensors!(A, B, buffer, offsets, ranks, rowpivots, colpivots)
    index = Int(blockIdx().x - 1) * blockDim().x + threadIdx().x - 1
    mr, maxM, nb = size(B)
    crosssize = mr * mr
    blocksize = crosssize + mr * maxM
    if index < blocksize * nb
        block = div(index, blocksize) + 1
        localindex = mod(index, blocksize)
        r = ranks[block]
        offset = offsets[block]
        if localindex < crosssize
            row = mod(localindex, mr) + 1
            col = div(localindex, mr) + 1
            @inbounds A[row, col, block] = if row <= r && col <= r
                buffer[offset + rowpivots[row, block], colpivots[col, block]]
            else
                (row == col ? one(eltype(A)) : zero(eltype(A)))
            end
        else
            localindex -= crosssize
            pivot = mod(localindex, mr) + 1
            row = div(localindex, mr) + 1
            @inbounds B[pivot, row, block] =
                if pivot <= r && row <= offsets[block + 1]-offset
                    buffer[offset + row, colpivots[pivot, block]]
                else
                    zero(eltype(B))
                end
        end
    end
    return nothing
end

# Build the batched (P_i, L_iᵀ) tensors from the pristine buffer and solve
# Pᵢᵀ Z = Lᵢᵀ for every block at once; returns host `Z` (maxrank×maxM×nblocks,
# padded) with block i's factor in `Z[1:r_i, 1:m_i, i]`.
function _solve_transposed_factors(
    stack::DistributedBlockStack, ranks, rowpivots, colpivots; maxrank=stack.maxrank
)
    T = eltype(stack.buffer)
    nb = nblocks(stack)
    mr = maxrank
    rowlens = [length(blockrows(stack, i)) for i in 1:nb]
    maxM = isempty(rowlens) ? 0 : maximum(rowlens)

    A = CuArray{T}(undef, mr, mr, max(nb, 1))
    B = CuArray{T}(undef, mr, max(maxM, 1), max(nb, 1))
    if nb == 0
        fill!(B, zero(T))
        return Array(B), maxM
    end
    offsets_d = CuArray(stack.rowoffsets)
    ranks_d = _deviceindices(ranks)
    rows_d = _deviceindices(rowpivots)
    cols_d = _deviceindices(colpivots)
    total = length(A) + length(B)
    @cuda threads=256 blocks=cld(total, 256) _gather_solve_tensors!(
        A, B, stack.buffer, offsets_d, ranks_d, rows_d, cols_d
    )

    # CUDA's batched wrappers pass raw pointers to the factor/pivot arrays.
    # Keep their owners alive through the solve and its synchronization.
    GC.@preserve A B begin
        pivarray, info, _ = getrf_strided_batched!(A, true)
        luinfo = Array(info)
        all(iszero, luinfo) || error("Singular GPU pivot matrix: LU info=$luinfo")
        GC.@preserve pivarray begin
            # cuBLAS getrsBatched crashes for these larger systems with the
            # device pointer mode used by CUDA.jl's task-local handle. Select
            # host pointer mode for this call and restore the caller's mode.
            handle = CUDA.CUBLAS.handle()
            mode = Ref{CUDA.CUBLAS.cublasPointerMode_t}()
            CUDA.CUBLAS.cublasGetPointerMode_v2(handle, mode)
            CUDA.CUBLAS.cublasSetPointerMode_v2(
                handle, CUDA.CUBLAS.CUBLAS_POINTER_MODE_HOST
            )
            try
                solveinfo, _ = getrs_strided_batched!('T', A, B, pivarray)
                solveinfo[] == 0 || error("GPU batched solve failed: info=$(solveinfo[])")
                return Array(B), maxM
            finally
                CUDA.CUBLAS.cublasSetPointerMode_v2(handle, mode[])
            end
        end
    end
end

# Each task owns one device's ACA copy, pivots, LU tensors and cuBLAS handle.
function _compress(cfg, stacks::AbstractVector{<:DistributedBlockStack})
    tasks = Task[]
    @sync for stack in stacks
        let stack=stack
            push!(tasks, Threads.@spawn _compress(cfg, stack))
        end
    end
    return fetch.(tasks)
end

# Host storage is updated after every device task has finished. k is
# shard-local (ranks/rowpivots/Z/blockrows); i is level-global (cluster metadata).
function _store_test!(
    stack,
    result,
    first,
    subtree,
    activeclusters,
    tvalues,
    childranges,
    nodechildren,
    tpivots,
    bases,
    transfer,
    pbar;
    scheduler=DynamicScheduler(),
)
    ranks, rowpivots, Z = result
    host_scheduler = if nblocks(stack) < 512
        SerialScheduler()
    else
        (scheduler === nothing ? DynamicScheduler() : scheduler)
    end
    @tasks for k in 1:nblocks(stack)
        @set scheduler = host_scheduler
        i = first - 1 + k
        t = activeclusters[i]
        r = ranks[k]
        m = length(blockrows(stack, k))
        tpivots[t] = tvalues[i][rowpivots[1:r, k]]
        Zi = @view Z[1:r, 1:m, k]                     # r×m (= Xᵢᵀ)
        if isleaf(subtree, t)
            bases[t] = Matrix(transpose(Zi))          # m×r
        else
            children = nodechildren[i]
            nodetransfers = Vector{Matrix{eltype(stack.buffer)}}(undef, length(children))
            for j in eachindex(children)
                cr = childranges[i][j]
                nodetransfers[j] = Matrix(transpose(@view Zi[:, cr]))
            end
            transfer[t] = nodetransfers
        end
        next!(pbar)
    end
    return nothing
end

function _store_trial!(
    stack,
    result,
    first,
    subtree,
    activeclusters,
    svalues,
    childranges,
    nodechildren,
    spivots,
    bases,
    transfer,
    pbar;
    scheduler=DynamicScheduler(),
)
    ranks, rowpivots, Z = result
    host_scheduler = if nblocks(stack) < 512
        SerialScheduler()
    else
        (scheduler === nothing ? DynamicScheduler() : scheduler)
    end
    @tasks for k in 1:nblocks(stack)
        @set scheduler = host_scheduler
        i = first - 1 + k
        s = activeclusters[i]
        r = ranks[k]
        m = length(blockrows(stack, k))
        spivots[s] = svalues[i][rowpivots[1:r, k]]
        Zi = @view Z[1:r, 1:m, k]                     # r×m (= Xᵢᵀ)
        if isleaf(subtree, s)
            bases[s] = Matrix(Zi)                     # r×m
        else
            children = nodechildren[i]
            nodetransfers = Vector{Matrix{eltype(stack.buffer)}}(undef, length(children))
            for j in eachindex(children)
                cr = childranges[i][j]
                nodetransfers[j] = Matrix(@view Zi[:, cr])
            end
            transfer[s] = nodetransfers
        end
        next!(pbar)
    end
    return nothing
end

function NestedCrossApproximation.compresslevel_test!(
    cfg,
    stacks::AbstractVector{<:DistributedBlockStack},
    subtree,
    activeclusters,
    tvalues,
    childranges,
    nodechildren,
    tpivots,
    bases,
    transfer,
    maxrank::Int,
    pbar;
    scheduler=nothing,  # GPU compression uses one task per device
)
    results = _compress(cfg, stacks)
    first = 1
    for (stack, result) in zip(stacks, results)
        _store_test!(
            stack,
            result,
            first,
            subtree,
            activeclusters,
            tvalues,
            childranges,
            nodechildren,
            tpivots,
            bases,
            transfer,
            pbar;
            scheduler,
        )
        first += nblocks(stack)
    end
    return nothing
end

function NestedCrossApproximation.compresslevel_trial!(
    cfg,
    stacks::AbstractVector{<:DistributedBlockStack},
    subtree,
    activeclusters,
    svalues,
    childranges,
    nodechildren,
    spivots,
    bases,
    transfer,
    maxrank::Int,
    pbar;
    scheduler=nothing,  # GPU compression uses one task per device
)
    results = _compress(cfg, stacks)
    first = 1
    for (stack, result) in zip(stacks, results)
        _store_trial!(
            stack,
            result,
            first,
            subtree,
            activeclusters,
            svalues,
            childranges,
            nodechildren,
            spivots,
            bases,
            transfer,
            pbar;
            scheduler,
        )
        first += nblocks(stack)
    end
    return nothing
end
