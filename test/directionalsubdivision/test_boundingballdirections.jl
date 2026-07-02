using Test
using NestedCrossApproximation
using LinearAlgebra
using StaticArrays

@testset "sphericalfibonaccipoints" begin
    n = 200
    pts = NestedCrossApproximation.sphericalfibonaccipoints(n)
    @test length(pts) == n
    @test all(p -> isapprox(norm(p), 1.0), pts)
    @test length(unique(pts)) == n
end

@testset "directions (2D)" begin
    F = Float64
    evecs = NestedCrossApproximation.directions(Val(2), F(1.0), F(1.0), F(1.0))
    @test length(evecs) >= 4
    @test all(e -> isapprox(norm(e), 1.0), evecs)

    # Finer (larger k*diam) admissibility needs at least as many directions.
    coarse = NestedCrossApproximation.directions(Val(2), F(1.0), F(0.1), F(1.0))
    fine = NestedCrossApproximation.directions(Val(2), F(1.0), F(10.0), F(1.0))
    @test length(fine) >= length(coarse)
end

@testset "directions (3D)" begin
    F = Float64
    evecs = NestedCrossApproximation.directions(Val(3), F(1.0), F(1.0), F(1.0))
    @test length(evecs) >= 6
    @test all(e -> isapprox(norm(e), 1.0), evecs)

    coarse = NestedCrossApproximation.directions(Val(3), F(1.0), F(0.1), F(1.0))
    fine = NestedCrossApproximation.directions(Val(3), F(1.0), F(10.0), F(1.0))
    @test length(fine) >= length(coarse)
end

@testset "_nearest_direction" begin
    evecs = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)]
    @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.9, 0.1, 0.1)) == 1
    @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.1, 0.9, 0.1)) == 2
    @test NestedCrossApproximation._nearest_direction(evecs, SVector(0.1, 0.1, 0.9)) == 3
end
