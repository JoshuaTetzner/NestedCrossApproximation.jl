using Test
using NestedCrossApproximation
using LinearAlgebra
using Random

@testset "nestedbasis.jl" begin
    scalartypes = (Float32, Float64, ComplexF32, ComplexF64)

    # Global row indices; the first length(pivots) columns belong to the cluster.
    @testset "nestedtestbasis ($T)" for T in scalartypes
        Random.seed!(1)
        buffer = rand(T, 12, 6)
        original = copy(buffer)
        tvalues = [2, 3, 5, 7, 8, 9, 11]
        pivots = [8, 3, 11]

        basis = @inferred NestedCrossApproximation.nestedtestbasis(tvalues, pivots, buffer)
        @test basis isa Matrix{T}
        @test size(basis) == (length(tvalues), length(pivots))
        @test buffer == original
        @test basis * buffer[pivots, 1:3] ≈ buffer[tvalues, 1:3]
        @test basis[indexin(pivots, tvalues), :] ≈ I

        buffer[:, 4:end] .= 0
        @test NestedCrossApproximation.nestedtestbasis(tvalues, pivots, buffer) == basis

        empty = NestedCrossApproximation.nestedtestbasis(tvalues, Int[], buffer)
        @test empty isa Matrix{T}
        @test size(empty) == (length(tvalues), 0)
    end

    @testset "nestedtrialbasis ($T)" for T in scalartypes
        Random.seed!(2)
        buffer = rand(T, 6, 12)
        original = copy(buffer)
        svalues = [2, 3, 5, 7, 8, 9, 11]
        pivots = [8, 3, 11]

        basis = @inferred NestedCrossApproximation.nestedtrialbasis(svalues, pivots, buffer)
        @test basis isa Matrix{T}
        @test size(basis) == (length(pivots), length(svalues))
        @test buffer == original
        @test buffer[1:3, pivots] * basis ≈ buffer[1:3, svalues]
        @test basis[:, indexin(pivots, svalues)] ≈ I

        buffer[4:end, :] .= 0
        @test NestedCrossApproximation.nestedtrialbasis(svalues, pivots, buffer) == basis

        empty = NestedCrossApproximation.nestedtrialbasis(svalues, Int[], buffer)
        @test empty isa Matrix{T}
        @test size(empty) == (0, length(svalues))
    end

    # Coincident points give identical rows, so a pivot set holding both is dependent.
    @testset "dependent pivots ($T)" for T in scalartypes
        Random.seed!(3)
        buffer = rand(T, 8, 2)
        buffer[7, :] = buffer[3, :]
        @test_throws NestedCrossApproximation.SingularCrossMatrixError NestedCrossApproximation.nestedtestbasis(
            collect(1:8), [3, 7], buffer
        )
        @test_throws NestedCrossApproximation.SingularCrossMatrixError NestedCrossApproximation.nestedtrialbasis(
            collect(1:8), [3, 7], permutedims(buffer)
        )
    end

    @testset "_build_basis_storage ($T)" for T in scalartypes
        bases = Vector{Matrix{T}}(undef, 5)
        bases[2] = rand(T, 4, 2)
        bases[5] = rand(T, 3, 1)

        store = @inferred NestedCrossApproximation._build_basis_storage(bases)
        @test store isa NestedCrossApproximation.BasisStore{
            T,NestedCrossApproximation.BasisTraversalPlan
        }
        @test store.plan.nodes == [2, 5]
        @test store.blocks[1] === bases[2]
        @test store.blocks[2] === bases[5]

        nobases = NestedCrossApproximation._build_basis_storage(Vector{Matrix{T}}(undef, 5))
        @test isempty(nobases.plan.nodes)
        @test nobases.blocks isa Vector{Matrix{T}}
        @test isempty(nobases.blocks)
    end

    # Three nodes own the slots 1:2, 3:3 and 4:6; `counts` is the number of bases per node.
    @testset "_build_dirbasis_storage ($T)" for T in scalartypes
        bases = Vector{Matrix{T}}(undef, 6)
        bases[1] = rand(T, 4, 2)
        bases[4] = rand(T, 3, 1)
        bases[6] = rand(T, 3, 2)
        counts = [1, 0, 2]

        store = @inferred NestedCrossApproximation._build_dirbasis_storage(bases, counts)
        @test store isa NestedCrossApproximation.BasisStore{
            T,NestedCrossApproximation.DirBasisTraversalPlan
        }
        @test store.plan.nodes == [1, 3]
        @test store.plan.idxtodiridxptr == [1, 2, 4]
        @test store.plan.diridx == [1, 4, 6]
        @test all(store.blocks[i] === bases[d] for (i, d) in enumerate(store.plan.diridx))

        nobases = NestedCrossApproximation._build_dirbasis_storage(
            Vector{Matrix{T}}(undef, 6), zeros(Int, 3)
        )
        @test isempty(nobases.plan.nodes)
        @test nobases.plan.idxtodiridxptr == [1]
        @test isempty(nobases.plan.diridx)
        @test isempty(nobases.blocks)
    end
end
