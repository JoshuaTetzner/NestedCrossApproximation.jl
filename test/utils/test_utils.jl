using Test
using NestedCrossApproximation
using LinearAlgebra

@testset "utils.jl" begin
    @testset "ptrfromcounts builds CSR offsets" begin
        # Entry i owns ptr[i]:ptr[i+1]-1, a slot exactly as wide as its count, and the
        # slots tile 1:sum(counts) without gap or overlap. A zero count is a zero-width
        # slot, not a skipped entry.
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

        # Nothing assigned: empty result, not an error - this is the shape the compressors
        # produce for a tree level that contributed no bases at all.
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

    # The guard that turns an unusable cross matrix into a diagnosis. Both nestedbasis.jl
    # and transfermatrices.jl divide by A[pivots, cols]; when the selected pivots are
    # linearly dependent that division is meaningless, and plain `lu!` reports it as a
    # bare SingularException from inside a parallel assembly.
    @testset "_crossfactorization" begin
        @testset "invertible cross matrix factorizes as usual" begin
            X = ComplexF64[2 1; 1 3]
            b = ComplexF64[1, 2]
            F = NestedCrossApproximation._crossfactorization(copy(X), :nested_test_basis)
            @test F isa LinearAlgebra.LU
            @test F \ b ≈ X \ b
        end

        @testset "dependent pivots raise SingularCrossMatrixError, not SingularException" begin
            # Two identical rows: exactly what coincident points produce once the pivot
            # search has picked a point and, later, its duplicate.
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
            # The message has to point at the matrix, not at the pivot search: the search
            # never repeats an index, so a dependent pivot set means duplicate rows.
            @test occursin("coincident points", msg)
        end
    end
end
