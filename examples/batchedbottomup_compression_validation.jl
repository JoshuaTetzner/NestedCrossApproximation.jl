# Validation harness for BatchedBottomUp's GPU compression path (complements the
# end-to-end timing in `batchedbottomup_gpu_vs_cpu.jl`).
#
# Two parts:
#   (A) CPU-only self-check of the flat-buffer ACA reference
#       (`ACACUDA.batched_fullpivot_aca_reference!`) - runs anywhere, no GPU.
#       Confirms the column-padded level-buffer layout and per-block pivot
#       extraction reproduce the single-block fully-pivoted ACA (CUR ~1e-13).
#   (B) End-to-end GPU-vs-CPU comparison of `testbases`/`trialbases` on the same
#       tree: asserts bit-for-bit identical pivots (deterministic tie-break) and
#       matching nested bases/transfer, and times each backend. Requires a
#       functional GPU; GPU assembly also runs ACA and LU on the GPU. With two
#       devices, compares one- and two-device ranks/pivots exactly and reports
#       bitwise basis/transfer equality as well as their numerical differences.
#
# Set `NCA_BATCHED_TIMING=1` for the per-level assemble/compress split.

using BEAST
using CompScienceMeshes
using ParallelKMeans
using H2Trees
using NestedCrossApproximation
using AdaptiveCrossApproximation
using CUDA
using Random
using LinearAlgebra

# ---------------------------------------------------------------------------
# (A) CPU self-check of the flat-buffer fully-pivoted ACA reference.
# ---------------------------------------------------------------------------
function validate_reference_cpu(; T=ComplexF64, maxrank=30, seed=1)
    Random.seed!(seed)
    acaext = Base.get_extension(AdaptiveCrossApproximation, :ACACUDA)
    acaext === nothing && error("ACACUDA extension not loaded (load CUDA first)")

    aca = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-12))

    # A handful of blocks of differing shape/rank, packed as contiguous row-slices
    # of one (Σrows × maxrank) column-padded buffer.
    specs = [(m=120, n=25, r=8), (m=64, n=30, r=17), (m=200, n=12, r=5), (m=90, n=20, r=20)]
    nb = length(specs)
    rowoffsets = zeros(Int, nb + 1)
    for i in 1:nb
        rowoffsets[i + 1] = rowoffsets[i] + specs[i].m
    end
    ncols = [s.n for s in specs]

    buffer = zeros(T, rowoffsets[end], maxrank)
    originals = Vector{Matrix{T}}(undef, nb)
    for i in 1:nb
        m, n, r = specs[i].m, specs[i].n, specs[i].r
        block = randn(T, m, r) * randn(T, r, n)     # exact rank r
        originals[i] = copy(block)
        buffer[(rowoffsets[i] + 1):rowoffsets[i + 1], 1:n] .= block
    end

    ranks, rowpivots, colpivots = acaext.batched_fullpivot_aca_reference!(
        aca, buffer, rowoffsets, ncols, maxrank
    )

    maxerr = 0.0
    for i in 1:nb
        r = ranks[i]
        rp = rowpivots[1:r, i]
        cp = colpivots[1:r, i]
        A = originals[i]
        recon = A[:, cp] * (A[rp, cp] \ A[rp, :])   # CUR from selected pivots
        err = norm(recon - A) / norm(A)
        maxerr = max(maxerr, err)
        println("  block $i: rank=$r (exact $(specs[i].r)), CUR relerr=$(err)")
    end
    println("(A) reference self-check max CUR relerr = ", maxerr)
    @assert maxerr < 1e-9 "flat-buffer reference failed CUR reconstruction"
    return nothing
end

# ---------------------------------------------------------------------------
# (B) End-to-end GPU-vs-CPU comparison on a real tree.
# ---------------------------------------------------------------------------
function compare_backends(; nrefine=8, minvalues=80, maxrank=30, device=0, devices=[device])
    Random.seed!(1)
    k = 2 * pi / 4.0
    Γ = meshicosphere(nrefine, 1.0)
    tRT = raviartthomas(Γ)
    sRT = raviartthomas(Γ)
    println("Size RT ", length(tRT))
    op = Maxwell3D.singlelayer(; wavenumber=k)

    tree = H2Trees.BlockTree(
        KMeansTree(
            tRT.pos;
            builder=KMeansTreeBuilder(; minvalues, updateradii=H2Trees.boundingsphere),
        ),
        KMeansTree(
            sRT.pos;
            builder=KMeansTreeBuilder(; minvalues, updateradii=H2Trees.boundingsphere),
        ),
    )
    testfardata, trialfardata = NestedCrossApproximation.fardata(
        tree, AdaptiveCrossApproximation.isnear()
    )

    farquadstrat = BEAST.DoubleNumQStrat(4, 5)
    cpumatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        op, tRT, sRT; matrixdata=farquadstrat
    )
    mktest() = NestedCrossApproximation.BatchedBottomUp(;
        pivotselector=TreeMimicryPivoting(tRT.pos, sRT.pos, H2Trees.trialtree(tree)),
        compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3)),
    )
    mktrial() = NestedCrossApproximation.BatchedBottomUp(;
        pivotselector=TreeMimicryPivoting(sRT.pos, tRT.pos, H2Trees.testtree(tree)),
        compressor=ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3)),
    )

    println("--- CPU testbases/trialbases ---")
    t_cpu = @elapsed begin
        cpu_tb, cpu_tt, cpu_tp = NestedCrossApproximation.testbases(
            mktest(), cpumatrix, tree, testfardata; maxrank=maxrank
        )
        cpu_sb, cpu_st, cpu_sp = NestedCrossApproximation.trialbases(
            mktrial(), cpumatrix, tree, trialfardata; maxrank=maxrank
        )
    end
    println("CPU total: ", round(t_cpu; digits=3), "s")

    isempty(devices) && throw(ArgumentError("at least one CUDA device is required"))
    single = nothing
    for selected in (length(devices) == 1 ? [devices] : [[devices[1]], devices])
        gpumatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
            op, tRT, sRT; matrixdata=GPUMatrixData(farquadstrat; devices=selected)
        )
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        # Empty levels and fewer blocks than devices must not create empty shards.
        @assert isempty(
            NestedCrossApproximation.assemble_level(
                gpumatrix, Vector{Int}[], Vector{Int}[]; maxrank
            ),
        )
        @assert CUDA.deviceid(CUDA.device()) == selected[1]
        shards = withenv("NCA_GPU_RESIDENT_COMPRESS" => "0") do
            return NestedCrossApproximation.assemble_level(
                gpumatrix, [collect(1:4)], [collect(100:103)]; maxrank
            )
        end
        @assert length(shards) == 1
        @assert only(shards) isa extension.DistributedBlockStack
        @assert only(shards).device == selected[1]
        # Three differently sized blocks exercise odd splits and shard ordering.
        rows = [collect(1:4), collect(5:10), collect(11:18)]
        cols = [collect(100:103), collect(110:114), collect(120:125)]
        shards = NestedCrossApproximation.assemble_level(gpumatrix, rows, cols; maxrank)
        ng = min(length(selected), length(rows))
        @assert length(shards) == ng
        @assert [stack.device for stack in shards] == selected[1:ng]
        @assert [extension.nblocks(stack) for stack in shards] == [div(g * 3, ng) - div((g-1) * 3, ng) for g in 1:ng]
        packed = Matrix{ComplexF64}[]
        for stack in shards
            CUDA.device!(stack.device)
            data = Array(stack.buffer)
            append!(
                packed,
                [
                    data[extension.blockrows(stack, i), 1:stack.ncols[i]] for
                    i in 1:extension.nblocks(stack)
                ],
            )
        end
        if single !== nothing
            @assert all(packed[i] == single.packed[i] for i in eachindex(packed))
        end
        CUDA.device!(selected[1])

        println("--- GPU testbases/trialbases: devices=$selected ---")
        # Host timing includes completion of every device task and host download.
        t_gpu = @elapsed begin
            gpu_tb, gpu_tt, gpu_tp = NestedCrossApproximation.testbases(
                mktest(), gpumatrix, tree, testfardata; maxrank
            )
            gpu_sb, gpu_st, gpu_sp = NestedCrossApproximation.trialbases(
                mktrial(), gpumatrix, tree, trialfardata; maxrank
            )
        end
        println("GPU total: ", round(t_gpu; digits=3), "s")
        @assert CUDA.deviceid(CUDA.device()) == selected[1]

        pivots_equal = _pivots_equal(cpu_tp, gpu_tp) && _pivots_equal(cpu_sp, gpu_sp)
        println("Pivots bit-for-bit equal (CPU vs GPU): ", pivots_equal)
        @assert pivots_equal "GPU ranks/pivots differ from CPU"
        berr = max(_store_maxreldiff(cpu_tb, gpu_tb), _store_maxreldiff(cpu_sb, gpu_sb))
        terr = max(_store_maxreldiff(cpu_tt, gpu_tt), _store_maxreldiff(cpu_st, gpu_st))
        println("CPU/GPU basis/transfer max relative differences: ", berr, " / ", terr)
        @assert berr < 1e-8 && terr < 1e-8 "GPU bases/transfer differ from CPU"

        if single === nothing
            single = (;
                tb=gpu_tb, tt=gpu_tt, tp=gpu_tp, sb=gpu_sb, st=gpu_st, sp=gpu_sp, packed
            )
        else
            @assert _pivots_equal(single.tp, gpu_tp) && _pivots_equal(single.sp, gpu_sp)
            berr = max(
                _store_maxreldiff(single.tb, gpu_tb), _store_maxreldiff(single.sb, gpu_sb)
            )
            terr = max(
                _store_maxreldiff(single.tt, gpu_tt), _store_maxreldiff(single.st, gpu_st)
            )
            bitwise_bases =
                _stores_equal(single.tb, gpu_tb) && _stores_equal(single.sb, gpu_sb)
            bitwise_transfers =
                _stores_equal(single.tt, gpu_tt) && _stores_equal(single.st, gpu_st)
            println(
                "One/multiple GPUs: ranks/pivots exactly equal; bases/transfers bitwise equal: ",
                bitwise_bases,
                " / ",
                bitwise_transfers,
            )
            println(
                "One/multiple GPU basis/transfer max relative differences: ",
                berr,
                " / ",
                terr,
            )
            # Different device hardware or cuBLAS batch shapes may round differently.
            @assert berr < 1e-8 && terr < 1e-8 "Multi-GPU bases/transfer differ from one GPU"
        end
        flush(stdout)
    end
    return nothing
end

_pivots_equal(a, b) =
    length(a) == length(b) && all(
        (isassigned(a, i) == isassigned(b, i)) && (!isassigned(a, i) || a[i] == b[i]) for
        i in eachindex(a)
    )

_stores_equal(a, b) =
    length(a.blocks) == length(b.blocks) &&
    all(isequal(x, y) for (x, y) in zip(a.blocks, b.blocks))

function _store_maxreldiff(cpu, gpu)
    cb, gb = cpu.blocks, gpu.blocks
    length(cb) == length(gb) || return Inf
    maxd = 0.0
    for i in eachindex(cb)
        size(cb[i]) == size(gb[i]) || return Inf
        maxd = max(maxd, norm(cb[i] - gb[i]) / max(norm(cb[i]), eps()))
    end
    return maxd
end

# ---------------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    validate_reference_cpu()
    if CUDA.functional()
        for nrefine in (8, 12)
            compare_backends(; nrefine, devices=collect(0:min(1, CUDA.ndevices() - 1)))
        end
    else
        println("\n[skip (B)] no functional CUDA device available.")
    end
end
