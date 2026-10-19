"""
    DistributedBlockStack{T}

Device-resident storage for one device's contiguous shard of a tree level,
packed into a column-padded buffer for batched ACA and basis solves.

  - `buffer::CuMatrix{T}`: `(Σ block-rows) × maxrank`. Block `i` occupies rows
    `rowoffsets[i]+1 : rowoffsets[i+1]` and columns `1:ncols[i]` (columns past
    `ncols[i]` are padding). This is exactly the flat layout the batched ACA
    (`ACACUDA`) and the batched-LU basis/transfer step consume.
  - `rowoffsets::Vector{Int}` (host): length `nblocks+1`, cumulative row offsets.
  - `ncols::Vector{Int}` (host): length `nblocks`, true column count per block.
  - `maxrank::Int`: buffer column width / per-block rank cap.
  - `device::Int`: CUDA device id the buffer lives on.

Returned in device order by `assemble_level(::GPUBEASTKernelMatrix, ...)`,
which returns a vector of stacks. The vector routes `compresslevel_*!` to the
GPU methods in `compresslevel.jl`. Indices within each stack are shard-local;
the prefix sum of their block counts gives their global positions in the level.
"""
struct DistributedBlockStack{T,B<:CuMatrix{T}}
    buffer::B
    rowoffsets::Vector{Int}
    ncols::Vector{Int}
    maxrank::Int
    device::Int
end

nblocks(stack::DistributedBlockStack) = length(stack.ncols)

# Row range (into `buffer`) and column count of block `i`.
@inline blockrows(stack::DistributedBlockStack, i::Int) =
    (stack.rowoffsets[i] + 1):stack.rowoffsets[i + 1]
@inline blockncols(stack::DistributedBlockStack, i::Int) = stack.ncols[i]

# GPU assembly keeps each shard on its device for batched ACA and basis solves.
# Each shard, its ACA working copy and its LU tensors must fit on its device.
function NestedCrossApproximation.assemble_level(
    farmatrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix{T},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    maxrank::Int,
    ntasks=nothing,
    transposed::Bool=false,
) where {T}
    return _assemble_level_resident(
        farmatrix, rowidcs, colidcs; maxrank=maxrank, transposed=transposed
    )
end

# Assemble independent contiguous shards without downloading the sampled blocks.
function _assemble_level_resident(
    farmatrix::AdaptiveCrossApproximation.GPUBEASTKernelMatrix{T},
    rowidcs::AbstractVector{<:AbstractVector{Int}},
    colidcs::AbstractVector{<:AbstractVector{Int}};
    maxrank::Int,
    transposed::Bool=false,
) where {T}
    length(rowidcs) == length(colidcs) ||
        throw(DimensionMismatch("row and column index collections must have equal length"))
    all(idcs -> length(idcs) <= maxrank, colidcs) ||
        throw(ArgumentError("a block has more columns than maxrank=$maxrank"))

    # Preserve the former main-task device for subsequent per-block assembly.
    CUDA.device!(farmatrix.functors[1].device)
    nb = length(rowidcs)
    ng = min(length(farmatrix.functors), nb)
    tasks = Task[]
    @sync for g in 1:ng
        idcs = (div((g - 1) * nb, ng) + 1):div(g * nb, ng)
        let functor=farmatrix.functors[g],
            rows=view(rowidcs, idcs),
            cols=view(colidcs, idcs)

            push!(
                tasks,
                Threads.@spawn begin
                    device = Int(functor.device)
                    CUDA.device!(device)
                    rowoffsets = [0; cumsum(length.(rows))]
                    ncols = length.(cols)
                    buffer = CUDA.zeros(T, rowoffsets[end], maxrank)
                    views = [
                        @view(buffer[(rowoffsets[i] + 1):rowoffsets[i + 1], 1:ncols[i]]) for
                        i in eachindex(rows)
                    ]
                    # Only this device's assembler may write these device views.
                    matrix = AdaptiveCrossApproximation.GPUBEASTKernelMatrix{T}([functor])
                    if transposed
                        # Assemble K[far test pivots, own trial values] directly
                        # into the transpose of each resident compression view.
                        AdaptiveCrossApproximation.assemble_blocks(
                            matrix, transpose.(views), cols, rows
                        )
                    else
                        AdaptiveCrossApproximation.assemble_blocks(
                            matrix, views, rows, cols
                        )
                    end
                    CUDA.synchronize(buffer)
                    DistributedBlockStack{T,typeof(buffer)}(
                        buffer, rowoffsets, ncols, maxrank, device
                    )
                end
            )
        end
    end
    # @sync waits for all tasks even when a shard fails. Keep the element type
    # explicit so the vector dispatch also works for an empty level.
    return DistributedBlockStack{T}[fetch(task) for task in tasks]
end
