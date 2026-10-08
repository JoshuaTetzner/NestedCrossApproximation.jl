using Test
using NestedCrossApproximation

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
end
