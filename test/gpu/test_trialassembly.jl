using Test
using BEAST
using CompScienceMeshes
using CUDA
using NestedCrossApproximation
using AdaptiveCrossApproximation
using LinearAlgebra

@testset "Resident transposed trial assembly on one and two GPUs" begin
    if CUDA.functional()
        CUDA.allowscalar(false)
        extension = Base.get_extension(
            NestedCrossApproximation, :NestedCrossApproximationCUDAExt
        )
        bext = Base.get_extension(BEAST, :BEASTCUDAExt)

        # Exercise the nested transpose-view scatter with genuinely complex data.
        CUDA.device!(0)
        buffer = CUDA.zeros(ComplexF64, 5, 3)
        destination = @view buffer[1:4, 1:2]
        output = CuArray(ComplexF64.(1:8) .+ im .* (11:18))
        worklist = (
            num_blocks=1, row_offsets=[0, 2], column_offsets=[0, 4], cell_offsets=[0, 8]
        )
        bext.scatter_batched_blocks!([transpose(destination)], output, worklist)
        @test Array(destination) == transpose(reshape(Array(output), 2, 4))
        @test iszero(Array(buffer)[5, 3])

        # Normal views and transposes of plain CuMatrices dispatch separately.
        normalworklist = merge(worklist, (row_offsets=[0, 4], column_offsets=[0, 2]))
        bext.scatter_batched_blocks!([destination], output, normalworklist)
        @test Array(destination) == reshape(Array(output), 4, 2)
        plain = CUDA.zeros(ComplexF64, 4, 2)
        bext.scatter_batched_blocks!([transpose(plain)], output, worklist)
        @test Array(plain) == transpose(reshape(Array(output), 2, 4))

        testspace = raviartthomas(meshicosphere(2, 1.0))
        trialmesh = translate!(meshicosphere(3, 0.8), point(6.0, 0.0, 0.0))
        trialspace = raviartthomas(trialmesh)
        @test length(testspace) == 120
        @test length(trialspace) == 270
        op = Maxwell3D.singlelayer(; wavenumber=2π/8)
        quadrature = BEAST.DoubleNumQStrat(2, 3)
        cpu = AbstractKernelMatrix(op, testspace, trialspace; matrixdata=quadrature)
        owntrial = [collect(241:270), collect(11:60), collect(70:149)]
        fartest = [collect(1:13), collect(77:91), collect(92:112)]
        expected = Matrix{ComplexF64}[]
        for (rows, cols) in zip(fartest, owntrial)
            block = zeros(ComplexF64, length(rows), length(cols))
            cpu(block, rows, cols)
            push!(expected, Matrix(transpose(block)))
        end
        configurations = CUDA.ndevices() >= 2 ? [[0], [0, 1], [1, 0]] : [[0]]
        cfg = ACA(FullPivoting(), FullPivoting(), FNormEstimator(1e-6))
        for devices in configurations
            gpu = AbstractKernelMatrix(
                op, testspace, trialspace; matrixdata=GPUMatrixData(quadrature; devices)
            )
            stacks = NestedCrossApproximation.assemble_level(
                gpu, owntrial, fartest; maxrank=30, transposed=true
            )
            @test length(stacks) == min(length(devices), length(expected))
            results = extension._compress(cfg, stacks)
            i = 0
            for (stack, (ranks, rowpivots, Z)) in zip(stacks, results)
                @test CUDA.deviceid(CUDA.device(stack.buffer)) == stack.device
                CUDA.device!(stack.device)
                for k in 1:extension.nblocks(stack)
                    i += 1
                    rows = extension.blockrows(stack, k)
                    block = Array(@view stack.buffer[rows, 1:stack.ncols[k]])
                    @test isapprox(block, expected[i]; rtol=1e-11, atol=1e-13)
                    rank, rp, _ = cfg(copy(block), 30; sampled=true)
                    @test ranks[k] == rank
                    @test rowpivots[1:rank, k] == rp
                    reconstructed = transpose(Z[1:rank, 1:length(rows), k]) * block[rp, :]
                    @test norm(reconstructed - block) / norm(block) < 1.01e-6
                end
            end
            @test i == length(expected)
        end
    else
        @test_skip CUDA.functional()
    end
end
