using Test
using NestedCrossApproximation
using H2Trees
using LinearAlgebra
using StaticArrays

@testset "abstractnca.jl" begin
    # Cluster g owns the coefficients ptr[g]:(ptr[g + 1] - 1), as many as it has pivots.
    @testset "plan_from_pivots" begin
        plan = @inferred NestedCrossApproximation.plan_from_pivots([
            [1, 2], [7, 8, 9], Int[], [4]
        ])
        @test plan isa NestedCrossApproximation.CoefficientPlan
        @test plan.globalidxtocoeffptr == [1, 3, 6, 6, 7]

        # A cluster without far field is unassigned and counts like one without pivots.
        sparsepivots = Vector{Vector{Int}}(undef, 4)
        sparsepivots[2] = [1, 2]
        sparsepivots[4] = Int[]
        @test NestedCrossApproximation.plan_from_pivots(sparsepivots).globalidxtocoeffptr ==
            [1, 1, 3, 3, 3]
        @test NestedCrossApproximation.plan_from_pivots(Vector{Vector{Int}}()).globalidxtocoeffptr ==
            [1]
    end

    @testset "assemblenears" begin
        testpts, trialpts = randomwellseparatedclouds(40; offset=SVector(0.5, 0.0, 0.0))
        tree = testblocktree(testpts, trialpts; minvalues=6)
        kernel = HelmholtzTestKernel(0.2)
        dense = densereference(kernel, testpts, trialpts)
        x, y = randn(ComplexF64, 40), randn(ComplexF64, 40)
        for isnear in
            ((args...) -> true, NestedCrossApproximation.isnear(2), (args...) -> false)
            partition = NestedCrossApproximation.interactions(tree, isnear)
            reference = zeros(ComplexF64, size(dense))
            for t in 1:(length(partition.nearptr) - 1)
                rows = H2Trees.values(H2Trees.testtree(tree), t)
                for k in partition.nearptr[t]:(partition.nearptr[t + 1] - 1)
                    cols = H2Trees.values(H2Trees.trialtree(tree), partition.nearnodes[k])
                    reference[rows, cols] = dense[rows, cols]
                end
            end
            for ntasks in (1, 2)
                near = NestedCrossApproximation.assemblenears(
                    kernel,
                    testpts,
                    trialpts,
                    tree;
                    isnear=(args...) -> error("The existing partition must be reused"),
                    partition=partition,
                    matrixdata=nothing,
                    ntasks=ntasks,
                )
                @test near * x ≈ reference * x
                @test adjoint(near) * y ≈ adjoint(reference) * y
            end
        end
    end
end
