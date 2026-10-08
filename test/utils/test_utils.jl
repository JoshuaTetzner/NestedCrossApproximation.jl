using Test
using NestedCrossApproximation
using LinearAlgebra

@testset "utils.jl" begin
    @testset "ptrfromcounts builds CSR offsets" begin
        # The slots tile 1:sum(counts); a zero count is a slot of zero width.
        counts = [2, 0, 3, 1]
        ptr = NestedCrossApproximation.ptrfromcounts(counts)
        @test ptr == [1, 3, 3, 6, 7]
        @test diff(ptr) == counts
        @test ptr[end] - 1 == sum(counts)

        # Offsets are Int whatever the counts were stored as.
        @test NestedCrossApproximation.ptrfromcounts(Int32[2, 1]) == [1, 3, 4]
        @test NestedCrossApproximation.ptrfromcounts(Int32[2, 1]) isa Vector{Int}

        # No entries at all still has to be indexable: consumers read ptr[i + 1].
        @test NestedCrossApproximation.ptrfromcounts(Int[]) == [1]
    end

    @testset "collect_assigned compacts a sparsely assigned vector" begin
        v = Vector{Vector{Int}}(undef, 5)
        v[2] = [1, 2]
        v[5] = [3]
        blocks, nodes = NestedCrossApproximation.collect_assigned(v)
        @test nodes == [2, 5]
        @test blocks == [[1, 2], [3]]

        # Nothing assigned gives an empty result.
        eblocks, enodes = NestedCrossApproximation.collect_assigned(
            Vector{Vector{Int}}(undef, 3)
        )
        @test isempty(eblocks)
        @test isempty(enodes)

        full = [[i] for i in 1:4]
        fblocks, fnodes = NestedCrossApproximation.collect_assigned(full)
        @test fnodes == 1:4
        @test fblocks == full
    end

    # Dependent pivots are reported as SingularCrossMatrixError instead of a bare
    # SingularException.
    @testset "_crossfactorization" begin
        @testset "invertible cross matrix factorizes as usual" begin
            X = ComplexF64[2 1; 1 3]
            b = ComplexF64[1, 2]
            F = NestedCrossApproximation._crossfactorization(copy(X), :nested_test_basis)
            @test F isa LinearAlgebra.LU
            @test F \ b ≈ X \ b
        end

        @testset "dependent pivots raise SingularCrossMatrixError, not SingularException" begin
            # Two identical rows, as coincident points produce them.
            S = ComplexF64[1 2; 1 2]
            err = try
                NestedCrossApproximation._crossfactorization(copy(S), :nested_test_basis)
                nothing
            catch e
                e
            end
            @test err isa NestedCrossApproximation.SingularCrossMatrixError
            @test err.npivots == 2
            @test err.rank == 1

            msg = sprint(showerror, err)
            @test occursin("nested_test_basis", msg)
            # The message points at the matrix, not at the pivot search.
            @test occursin("coincident points", msg)
        end
    end
end
