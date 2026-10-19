using Test
using NestedCrossApproximation
using AdaptiveCrossApproximation
using H2Trees
using ParallelKMeans
using StaticArrays
using OhMyThreads

struct SchedulerProbeKernel
    assembly_ntasks::Vector{Any}
end
Base.eltype(::SchedulerProbeKernel) = Float64
(kernel::SchedulerProbeKernel)(x, y) = exp(-sum(abs2, x - y))

function AdaptiveCrossApproximation.assemble_blocks(
    matrix::AdaptiveCrossApproximation.PointMatrix{T,SchedulerProbeKernel},
    blocks::AbstractVector{<:AbstractMatrix{T}},
    rows::AbstractVector{<:AbstractVector{Int}},
    cols::AbstractVector{<:AbstractVector{Int}};
    ntasks=1,
    kwargs...,
) where {T}
    push!(matrix.functor.assembly_ntasks, ntasks)
    return invoke(
        AdaptiveCrossApproximation.assemble_blocks,
        Tuple{
            AdaptiveCrossApproximation.AbstractKernelMatrix{T},
            AbstractVector{<:AbstractMatrix{T}},
            AbstractVector{<:AbstractVector{Int}},
            AbstractVector{<:AbstractVector{Int}},
        },
        matrix,
        blocks,
        rows,
        cols;
        ntasks,
        kwargs...,
    )
end

@testset "ntasks of assembly and matrix-vector product" begin
    points = [SVector(i / 15, j / 15, 0.0) for i in 0:15 for j in 0:15]
    cluster = KMeansTree(points; builder=KMeansTreeBuilder(; minvalues=80))
    tree = BlockTree(cluster, cluster)
    kernel = SchedulerProbeKernel(Any[])

    serial = NestedCrossApproximation.PetrovGalerkinNCA(
        kernel, points, points, tree; ntasks=1
    )
    @test !isempty(kernel.assembly_ntasks)
    @test all(==(1), kernel.assembly_ntasks)
    @test serial.scheduler isa SerialScheduler
    nears = serial.nearinteractions
    @test nears.scheduler isa SerialScheduler

    empty!(kernel.assembly_ntasks)
    parallel = NestedCrossApproximation.PetrovGalerkinNCA(
        kernel, points, points, tree; ntasks=2
    )
    @test all(==(2), kernel.assembly_ntasks)
    @test parallel.scheduler isa DynamicScheduler
    @test nears.blocks == parallel.nearinteractions.blocks
    @test nears.rowindices == parallel.nearinteractions.rowindices
    @test nears.colindices == parallel.nearinteractions.colindices

    x = randn(length(points))
    @test serial * x ≈ parallel * x
end
