using Test
using NestedCrossApproximation
using H2Trees
using AdaptiveCrossApproximation
using ParallelKMeans
using StaticArrays
using Random
using LinearAlgebra
using OhMyThreads

include(joinpath(@__DIR__, "testutils.jl"))

@testset "MVP threading (parallel vs serial)" begin
    N = 500
    testpts, trialpts = randomwellseparatedclouds(N; seed=42)
    kernel = HelmholtzTestKernel(2.0)
    tree = testblocktree(testpts, trialpts; minvalues=8)
    A = densereference(kernel, testpts, trialpts)

    Random.seed!(11)
    x = rand(ComplexF64, N)
    y_dense = A * x

    # BottomUp handles all three wideband regimes; TopDown only supports FarData
    compressor = NestedCrossApproximation.BottomUp(; factorization=ACA())

    # Three frequency regimes: wavenumbers chosen to produce all-LF, genuine
    # mixed LF/HF, and all-HF interactions for a KMeansTree (same thresholds
    # verified in test_directionaldata.jl for BoundingBallTree)
    isnear_regimes = [
        "all-LF" => NestedCrossApproximation.isnearwideband(2π / 50.0),
        "mixed LF/HF" => NestedCrossApproximation.isnearwideband(2π / 2.5),
        "all-HF" => NestedCrossApproximation.isnearwideband(2π / 0.02),
    ]

    @testset "forward MVP: accuracy and parallel==serial" begin
        for (name, isnear) in isnear_regimes
            @testset "$name" begin
                hmat = NestedCrossApproximation.PetrovGalerkinNCA(
                    kernel,
                    testpts,
                    trialpts,
                    tree;
                    testcompressor=compressor,
                    trialcompressor=compressor,
                    isnear=isnear,
                    scheduler=SerialScheduler(),
                    maxrank=25,
                )
                y_serial = hmat * x
                @test norm(y_dense - y_serial) / norm(y_dense) < 1e-2

                hmat_par = NestedCrossApproximation.PetrovGalerkinNCA{ComplexF64}(
                    hmat.tree,
                    hmat.nearinteractions,
                    hmat.nestedtestbases,
                    hmat.nestedtrialbases,
                    hmat.testtransfermatrices,
                    hmat.trialtransfermatrices,
                    hmat.couplingmatrices,
                    hmat.aggregationplan,
                    hmat.disaggregationplan,
                    DynamicScheduler(),
                    hmat.dim,
                )
                for _ in 1:20
                    @test hmat_par * x ≈ y_serial atol=0.0
                end
            end
        end
    end

    # Isolation tests use the mixed regime: exercises both LF and HF code paths
    mixed_isnear = NestedCrossApproximation.isnearwideband(2π / 2.5)

    @testset "aggregate_coefficients! thread-safe" begin
        hmat = NestedCrossApproximation.PetrovGalerkinNCA(
            kernel,
            testpts,
            trialpts,
            tree;
            testcompressor=compressor,
            trialcompressor=compressor,
            isnear=mixed_isnear,
            scheduler=SerialScheduler(),
            maxrank=25,
        )
        T = eltype(hmat)
        n_agg = hmat.aggregationplan.globalidxtocoeffptr[end] - 1
        xhat0 = zeros(T, n_agg)
        NestedCrossApproximation._project_to_coefficients!(
            xhat0,
            x,
            hmat.nestedtrialbases,
            hmat.aggregationplan,
            node -> H2Trees.values(H2Trees.trialtree(hmat.tree), node),
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        xhat_s = copy(xhat0)
        NestedCrossApproximation._aggregate_coefficients!(
            xhat_s,
            hmat.trialtransfermatrices,
            hmat.aggregationplan,
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        for _ in 1:20
            xhat_p = copy(xhat0)
            NestedCrossApproximation._aggregate_coefficients!(
                xhat_p,
                hmat.trialtransfermatrices,
                hmat.aggregationplan,
                NestedCrossApproximation._idop,
                DynamicScheduler(),
            )
            @test xhat_p == xhat_s
        end
    end

    @testset "disaggregate_coefficients! thread-safe" begin
        hmat = NestedCrossApproximation.PetrovGalerkinNCA(
            kernel,
            testpts,
            trialpts,
            tree;
            testcompressor=compressor,
            trialcompressor=compressor,
            isnear=mixed_isnear,
            scheduler=SerialScheduler(),
            maxrank=25,
        )
        T = eltype(hmat)
        n_agg = hmat.aggregationplan.globalidxtocoeffptr[end] - 1
        n_dis = hmat.disaggregationplan.globalidxtocoeffptr[end] - 1
        xhat0 = zeros(T, n_agg)
        NestedCrossApproximation._project_to_coefficients!(
            xhat0,
            x,
            hmat.nestedtrialbases,
            hmat.aggregationplan,
            node -> H2Trees.values(H2Trees.trialtree(hmat.tree), node),
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        xhat_s = copy(xhat0)
        NestedCrossApproximation._aggregate_coefficients!(
            xhat_s,
            hmat.trialtransfermatrices,
            hmat.aggregationplan,
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        yhat_s = zeros(T, n_dis)
        NestedCrossApproximation._couple_forward!(
            yhat_s,
            xhat_s,
            hmat.couplingmatrices,
            hmat.disaggregationplan,
            hmat.aggregationplan,
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        yhat_dis_s = copy(yhat_s)
        NestedCrossApproximation._disaggregate_coefficients!(
            yhat_dis_s,
            hmat.testtransfermatrices,
            hmat.disaggregationplan,
            NestedCrossApproximation._idop,
            SerialScheduler(),
        )
        for _ in 1:20
            yhat_dis_p = copy(yhat_s)
            NestedCrossApproximation._disaggregate_coefficients!(
                yhat_dis_p,
                hmat.testtransfermatrices,
                hmat.disaggregationplan,
                NestedCrossApproximation._idop,
                DynamicScheduler(),
            )
            @test yhat_dis_p == yhat_dis_s
        end
    end

    @testset "parallel assembly reproducibility (same tree)" begin
        for (name, isnear) in isnear_regimes
            @testset "$name" begin
                ys = map(1:5) do _
                    h = NestedCrossApproximation.PetrovGalerkinNCA(
                        kernel,
                        testpts,
                        trialpts,
                        tree;
                        testcompressor=compressor,
                        trialcompressor=compressor,
                        isnear=isnear,
                        scheduler=DynamicScheduler(),
                        maxrank=25,
                    )
                    return h * x
                end
                for i in 2:5
                    @test ys[i] == ys[1]
                end
            end
        end
    end
end
