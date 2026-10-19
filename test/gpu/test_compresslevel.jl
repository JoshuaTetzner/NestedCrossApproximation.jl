using Test
using CUDA
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra
using Random
using H2Trees
using OhMyThreads

struct GPUStorageTestTree end
H2Trees.isleaf(::GPUStorageTestTree, node::Int) = isodd(node)

@testset "Parallel GPU basis storage preserves shard ordering" begin
    if CUDA.functional()
        CUDA.device!(0)
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        nb, mr, first_block = 640, 4, 3
        lengths = [isodd(k) ? 6 : 8 for k in 1:nb]
        offsets = [0; cumsum(lengths)]
        stack = extension.DistributedBlockStack(
            CUDA.zeros(ComplexF64, offsets[end], mr), offsets, fill(mr, nb), mr, 0
        )
        ranks = [mod(k, mr + 1) for k in 1:nb]
        rowpivots = repeat(collect(1:mr), 1, nb)
        Z = randn(MersenneTwister(71), ComplexF64, mr, maximum(lengths), nb)
        nodes = collect(1:(nb + 4))
        values = [collect(1:8) for _ in nodes]
        ranges = [[1:3, 4:8] for _ in nodes]
        children = [[1, 2] for _ in nodes]
        for k in 1:nb
            ranges[first_block - 1 + k] = [1:3, 4:lengths[k]]
        end
        for store! in (extension._store_test!, extension._store_trial!)
            function stored(scheduler)
                pivots = [Int[] for _ in nodes]
                bases = [zeros(ComplexF64, 0, 0) for _ in nodes]
                transfers = [Matrix{ComplexF64}[] for _ in nodes]
                pbar = NestedCrossApproximation.Progress(0; enabled=false)
                store!(
                    stack,
                    (ranks, rowpivots, Z),
                    first_block,
                    GPUStorageTestTree(),
                    nodes,
                    values,
                    ranges,
                    children,
                    pivots,
                    bases,
                    transfers,
                    pbar;
                    scheduler,
                )
                return pivots, bases, transfers
            end
            @test stored(DynamicScheduler(; ntasks=8)) == stored(SerialScheduler())
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "Resident GPU basis solve above rank 32" begin
    if CUDA.functional()
        device = parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0"))
        CUDA.device!(device)
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        @test extension !== nothing
        # Rank 80 / 160 RHS reproduced a native crash in getrsBatched with the
        # default device pointer mode; complex data also checks transpose vs adjoint.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            rng = MersenneTwister(17)
            mr = 80
            lengths = [40, 90, 100, 160]
            ranks = [0, 8, 35, 80]
            offsets = [0; cumsum(lengths)]
            buffer = zeros(T, offsets[end], mr)
            rowpivots = zeros(Int, mr, length(lengths))
            colpivots = zeros(Int, mr, length(lengths))
            expected = Matrix{T}[]
            for i in eachindex(lengths)
                m, r = lengths[i], ranks[i]
                if r == 0
                    push!(expected, zeros(T, 0, m))
                    continue
                end
                rows = randperm(rng, m)[1:r]
                cols = randperm(rng, mr)[1:r]
                L = randn(rng, T, m, r)
                L[rows, :] += r * Matrix{T}(I, r, r)
                buffer[(offsets[i] + 1):offsets[i + 1], cols] = L
                rowpivots[1:r, i] = rows
                colpivots[1:r, i] = cols
                push!(expected, Matrix(transpose(L / L[rows, :])))
            end
            stack = extension.DistributedBlockStack(
                CuArray(buffer), offsets, fill(mr, length(lengths)), mr, device
            )
            mode_before = Ref{CUDA.CUBLAS.cublasPointerMode_t}()
            CUDA.CUBLAS.cublasGetPointerMode_v2(CUDA.CUBLAS.handle(), mode_before)
            Z, maxM = extension._solve_transposed_factors(
                stack, ranks, rowpivots, colpivots
            )
            @test maxM == maximum(lengths)
            tolerance = real(T) === Float32 ? 1e-4 : 1e-11
            for i in eachindex(lengths)
                @test isapprox(
                    Z[1:ranks[i], 1:lengths[i], i],
                    expected[i];
                    rtol=tolerance,
                    atol=tolerance,
                )
            end
            mode_after = Ref{CUDA.CUBLAS.cublasPointerMode_t}()
            CUDA.CUBLAS.cublasGetPointerMode_v2(CUDA.CUBLAS.handle(), mode_after)
            @test mode_after[] == mode_before[]
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "Resident GPU ACA residual and CPU agreement" begin
    if CUDA.functional()
        device = parse(Int, get(ENV, "BEAST_GPU_DEVICE", "0"))
        CUDA.device!(device)
        rng = MersenneTwister(41)
        mr = 30
        specs = [(40, 25), (90, 30), (65, 18)]
        offsets = [0; cumsum(first.(specs))]
        # Include a zero block and decaying, full-rank complex blocks. These
        # exercise stopping at the requested tolerance rather than the rank cap.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            originals = Matrix{T}[]
            buffer = zeros(T, offsets[end], mr)
            for (i, (m, n)) in enumerate(specs)
                U = Matrix(qr(randn(rng, T, m, n)).Q)[:, 1:n]
                V = Matrix(qr(randn(rng, T, n, n)).Q)
                block = if i == 1
                    zeros(T, m, n)
                else
                    U * Diagonal(real(T).(10.0 .^ range(0, -7; length=n))) * V'
                end
                push!(originals, block)
                buffer[(offsets[i] + 1):offsets[i + 1], 1:n] = block
            end
            tol = real(T)(1e-3)
            cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(tol))
            ranks_d, rows_d, cols_d = cfg(
                CuArray(buffer), offsets, last.(specs), mr; sampled=true
            )
            ranks, rows, cols = Int.(Array(ranks_d)),
            Int.(Array(rows_d)),
            Int.(Array(cols_d))
            extension = Base.get_extension(
                NestedCrossApproximation, :NestedCrossApproximationCUDAExt
            )
            stack = extension.DistributedBlockStack(
                CuArray(buffer), offsets, last.(specs), mr, device
            )
            resident_ranks, resident_rows, Z = extension._compress(cfg, stack)
            @test resident_ranks == ranks
            @test resident_rows == rows
            @test size(Z, 1) == max(1, maximum(ranks))
            for (i, block) in enumerate(originals)
                rank, rp, cp = cfg(copy(block), mr; sampled=true)
                @test ranks[i] == rank
                @test rows[1:rank, i] == rp
                @test cols[1:rank, i] == cp
                if rank == 0
                    @test iszero(norm(block))
                    continue
                end
                reconstructed = transpose(Z[1:rank, 1:size(block, 1), i]) * block[rp, :]
                @test rank < min(size(block)...)
                m, n = size(block)
                estimated_error =
                    norm(reconstructed - block) / norm(block) * sqrt(n / (n - rank))
                @test estimated_error <= 1.01tol
            end
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "GPU solve padding for zero-rank blocks" begin
    if CUDA.functional()
        CUDA.device!(0)
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3))
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            stack = extension.DistributedBlockStack(
                CUDA.zeros(T, 13, 30), [0, 0, 4, 13], [3, 3, 3], 30, 0
            )
            ranks, rows, Z = extension._compress(cfg, stack)
            @test ranks == [0, 0, 0]
            @test size(rows) == (30, 3)
            @test size(Z) == (1, 9, 3)
            @test iszero(norm(Z))
        end
    else
        @test_skip CUDA.functional()
    end
end

@testset "Concurrent resident compression on two GPUs" begin
    if CUDA.functional() && CUDA.ndevices() >= 2
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        devices = [0, 1]
        mr = 80
        lengths, expected_ranks = [40, 90, 100, 160], [0, 8, 35, 80]
        offsets = [0; cumsum(lengths)]
        # Rank >32 and varying RHS counts exercise the cuBLAS pointer-mode
        # workaround under concurrency, with real and complex precision variants.
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            rng = MersenneTwister(71)
            buffer = zeros(T, offsets[end], mr)
            for (i, (m, r)) in enumerate(zip(lengths, expected_ranks))
                r == 0 && continue
                L = randn(rng, T, m, r)
                L[1:r, :] += r * Matrix{T}(I, r, r)
                buffer[(offsets[i] + 1):offsets[i + 1], 1:r] = L
            end
            cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(real(T)(1e-5)))
            CUDA.device!(devices[1])
            single = extension.DistributedBlockStack(
                CuArray(buffer), offsets, fill(mr, 4), mr, devices[1]
            )
            expected = extension._compress(cfg, single)
            @test expected[1] == expected_ranks
            shards = extension.DistributedBlockStack{T}[]
            for (g, idcs) in enumerate((1:2, 3:4))
                CUDA.device!(devices[g])
                rows = (offsets[first(idcs)] + 1):offsets[last(idcs) + 1]
                localoffsets = [0; cumsum(lengths[idcs])]
                push!(
                    shards,
                    extension.DistributedBlockStack(
                        CuArray(buffer[rows, :]),
                        localoffsets,
                        fill(mr, length(idcs)),
                        mr,
                        devices[g],
                    ),
                )
            end
            CUDA.device!(devices[1])
            results = extension._compress(cfg, shards)
            @test CUDA.deviceid(CUDA.device()) == devices[1]
            @test vcat((result[1] for result in results)...) == expected[1]
            @test hcat((result[2] for result in results)...) == expected[2]
            tolerance = real(T) === Float32 ? 1e-4 : 1e-11
            firstblock = 1
            for (stack, (ranks, rows, Z)) in zip(shards, results)
                for k in eachindex(ranks)
                    i = firstblock + k - 1
                    r, m = ranks[k], lengths[i]
                    @test isapprox(
                        Z[1:r, 1:m, k],
                        expected[3][1:r, 1:m, i];
                        rtol=tolerance,
                        atol=tolerance,
                    )
                end
                firstblock += extension.nblocks(stack)
            end
        end
        @test isempty(
            extension._compress(
                ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3)),
                extension.DistributedBlockStack{Float64}[],
            ),
        )
    else
        @test_skip CUDA.functional() && CUDA.ndevices() >= 2
    end
end

@testset "Multi-GPU device state and recovery after worker failures" begin
    if CUDA.functional() && CUDA.ndevices() >= 2
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        devices = reverse(collect(0:min(3, CUDA.ndevices() - 1)))
        cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-3))
        rng = MersenneTwister(91)
        originals = [
            randn(rng, ComplexF64, 40 + 5g, 12) * randn(rng, ComplexF64, 12, 30) for
            g in eachindex(devices)
        ]
        originals[1][:, end] .= 0
        shards = extension.DistributedBlockStack{ComplexF64}[]
        for (g, dev) in enumerate(devices)
            CUDA.device!(dev)
            block = originals[g]
            push!(
                shards,
                extension.DistributedBlockStack(
                    CuArray(block), [0, size(block, 1)], [30], 30, dev
                ),
            )
        end
        CUDA.device!(0)
        records = Channel{Any}(length(devices))
        checkedcfg = function (buffer, offsets, ncols, maxrank; sampled)
            handle = CUDA.CUBLAS.handle()
            mode = Ref{CUDA.CUBLAS.cublasPointerMode_t}()
            CUDA.CUBLAS.cublasGetPointerMode_v2(handle, mode)
            put!(
                records,
                (;
                    task=current_task(),
                    handle,
                    mode=mode[],
                    device=CUDA.deviceid(CUDA.device()),
                    bufferdevice=CUDA.deviceid(CUDA.device(buffer)),
                ),
            )
            return cfg(buffer, offsets, ncols, maxrank; sampled)
        end
        results = extension._compress(checkedcfg, shards)
        @test CUDA.deviceid(CUDA.device()) == 0
        workers = [take!(records) for _ in devices]
        @test all(worker -> istaskdone(worker.task), workers)
        @test length(unique(worker.handle for worker in workers)) == length(devices)
        @test sort([worker.device for worker in workers]) == sort(devices)
        for worker in workers
            @test worker.bufferdevice == worker.device
            CUDA.device!(worker.device)
            mode = Ref{CUDA.CUBLAS.cublasPointerMode_t}()
            CUDA.CUBLAS.cublasGetPointerMode_v2(worker.handle, mode)
            @test mode[] == worker.mode
        end

        # A host-side error must wait for every other worker before escaping.
        # The channel handshake guarantees another worker has started first.
        ready = Channel{Int}(length(devices))
        tasks = Channel{Task}(length(devices))
        failcfg = function (buffer, offsets, ncols, maxrank; sampled)
            put!(tasks, current_task())
            dev = CUDA.deviceid(CUDA.device())
            if dev == devices[1]
                take!(ready)
                error("intentional compression worker failure")
            end
            put!(ready, dev)
            return cfg(buffer, offsets, ncols, maxrank; sampled)
        end
        CUDA.device!(0)
        @test_throws CompositeException extension._compress(failcfg, shards)
        @test all(istaskdone, [take!(tasks) for _ in devices])
        @test CUDA.deviceid(CUDA.device()) == 0

        # Select the known zero column to test a numerical LU failure,
        # without a CUDA fault, then verify a normal compression still succeeds.
        tasks = Channel{Task}(length(devices))
        singularcfg = function (buffer, offsets, ncols, maxrank; sampled)
            put!(tasks, current_task())
            ranks, rows, cols = cfg(buffer, offsets, ncols, maxrank; sampled)
            if CUDA.deviceid(CUDA.device()) == devices[1]
                @views cols[2:2, :] .= Int32(30)
            end
            return ranks, rows, cols
        end
        failure = try
            extension._compress(singularcfg, shards)
            nothing
        catch err
            err
        end
        @test failure isa CompositeException
        @test occursin("Singular GPU pivot matrix", sprint(showerror, failure))
        @test all(istaskdone, [take!(tasks) for _ in devices])
        @test CUDA.deviceid(CUDA.device()) == 0
        recovered = extension._compress(cfg, shards)
        for (g, stack) in enumerate(shards)
            @test recovered[g][1] == results[g][1]
            @test recovered[g][2] == results[g][2]
            @test recovered[g][3] == results[g][3]
            CUDA.device!(stack.device)
            @test Array(stack.buffer) == originals[g]
        end
        CUDA.device!(0)
    else
        @test_skip CUDA.functional() && CUDA.ndevices() >= 2
    end
end
