"""
    PetrovGalerkinNCA{T} <: LinearMaps.LinearMap{T}

H²-matrix constructed with the nested cross approximation.

Near interactions are stored as dense blocks. Far interactions are stored in nested form:
a basis for each leaf cluster, a transfer matrix for each cluster and child, and a coupling
matrix for each admissible pair of clusters. Test and trial side have separate bases, so
the test and the trial space may differ.

# Fields

  - `tree`: block tree of the test and the trial cluster tree
  - `nearinteractions`: block-sparse matrix of the near field
  - `nestedtestbases`, `nestedtrialbases`: leaf bases, see [`BasisStore`](@ref)
  - `testtransfermatrices`, `trialtransfermatrices`: transfer matrices, see
    [`TransferStore`](@ref)
  - `couplingmatrices`: coupling matrices, see [`CouplingStore`](@ref)
  - `aggregationplan`, `disaggregationplan`: position of the coefficients of each trial and
    test cluster in the coefficient vectors of the matrix-vector product
  - `scheduler`: scheduler of the matrix-vector product, created from `ntasks` by the
    constructor
  - `dim::Tuple{Int,Int}`: matrix dimensions `(m, n)`

# Notes

A `PetrovGalerkinNCA` is a linear map that supports `*`, `mul!`, `transpose`, and `adjoint`.
It is created through the constructor
`PetrovGalerkinNCA(operator, testspace, trialspace, tree; kwargs...)`.

# See also

[`farmatrix`](@ref), [`nearmatrix`](@ref), [`storage`](@ref)
"""
struct PetrovGalerkinNCA{
    T,
    TreeType,
    NearInteractionType,
    NestedBasesDict,
    TransferMatrixType,
    CouplingMatrixType,
    PlanType,
    SchedulerType,
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    aggregationplan::PlanType
    disaggregationplan::PlanType
    scheduler::SchedulerType
    dim::Tuple{Int,Int}

    function PetrovGalerkinNCA{T}(
        tree,
        nearinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        aggregationplan,
        disaggregationplan,
        scheduler,
        dim,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
            typeof(aggregationplan),
            typeof(scheduler),
        }(
            tree,
            nearinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            aggregationplan,
            disaggregationplan,
            scheduler,
            dim,
        )
    end
end

"""
    PetrovGalerkinNCA(operator, testspace, trialspace, tree; kwargs...)

Assemble an H²-matrix of `operator` with the nested cross approximation.

The near field is assembled densely. For the far field the test and the trial compressor
select pivots for every cluster, from which the nested bases, the transfer matrices and
the coupling matrices are computed.

# Arguments

  - `operator`: operator or kernel that defines the matrix entries
  - `testspace`, `trialspace`: test and trial space, e.g. basis functions or point sets.
    Together with `operator` they have to be supported by
    `AdaptiveCrossApproximation.AbstractKernelMatrix`.
  - `tree`: block tree (`H2Trees.BlockTree`) of the cluster trees of both spaces

# Keyword Arguments

  - `testcompressor = TopDown()`: compressor for the test bases, one of [`TopDown`](@ref),
    [`BottomUp`](@ref), and [`BatchedBottomUp`](@ref)
  - `trialcompressor = TopDown()`: compressor for the trial bases
  - `isnear = isnear()`: admissibility condition. With [`isnearwideband`](@ref) a directional
    H²-matrix is built, which requires [`BottomUp`](@ref) on both sides.
  - `maxrank::Int = 50`: maximum number of pivots of a cluster
  - `matrixdata`: data for the evaluation of the near field, e.g. a quadrature strategy
  - `farmatrixdata`: data for the evaluation of the far field
  - `ntasks::Integer = Threads.nthreads()`: number of parallel tasks for the assembly and
    the matrix-vector product; `ntasks = 1` runs on the calling task
  - `verbose::Bool = false`: show progress bars during the assembly

# Returns

  - [`PetrovGalerkinNCA`](@ref)
"""
function PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace,
    tree;
    matrixdata=AdaptiveCrossApproximation.defaultmatrixdata(
        operator, testspace, trialspace
    ),
    farmatrixdata=AdaptiveCrossApproximation.defaultfarmatrixdata(
        operator, testspace, trialspace
    ),
    testcompressor=TopDown(),
    trialcompressor=TopDown(),
    ntasks::Integer=Threads.nthreads(),
    isnear=isnear(),
    maxrank=50,
    verbose::Bool=false,
)
    partition = interactions(tree, isnear)
    nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        ntasks=ntasks,
        matrixdata=matrixdata,
        partition=partition,
        verbose=verbose,
    )

    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=farmatrixdata
    )
    fars = assemblefars(
        tree,
        isnear,
        testcompressor,
        trialcompressor,
        farmatrix;
        ntasks=ntasks,
        maxrank=maxrank,
        partition=partition,
        verbose=verbose,
    )
    aggregationplan = plan_from_pivots(fars.trialpivots)
    disaggregationplan = plan_from_pivots(fars.testpivots)

    return PetrovGalerkinNCA{eltype(farmatrix)}(
        tree,
        nears,
        fars.nestedtestbases,
        fars.nestedtrialbases,
        fars.testtransfermats,
        fars.trialtransfermats,
        fars.couplingmatrices,
        aggregationplan,
        disaggregationplan,
        _scheduler(ntasks),
        (length(testspace), length(trialspace)),
    )
end

Base.size(A::PetrovGalerkinNCA) = A.dim
Base.size(A::PetrovGalerkinNCA, dim::Int) = A.dim[dim]
Base.eltype(::PetrovGalerkinNCA{T}) where {T} = T

"""
    farmatrix(h2mat::PetrovGalerkinNCA)

Return the far field of `h2mat` as a [`PetrovGalerkinNCA`](@ref) with an empty near field.

The nested bases, transfer and coupling matrices are shared with `h2mat`.
"""
function farmatrix(h2mat::PetrovGalerkinNCA)
    blocks = Matrix{eltype(h2mat)}[]
    nears = BlockSparseMatrix(blocks, Vector{Int}[], Vector{Int}[], h2mat.dim)

    return PetrovGalerkinNCA{eltype(h2mat)}(
        h2mat.tree,
        nears,
        h2mat.nestedtestbases,
        h2mat.nestedtrialbases,
        h2mat.testtransfermatrices,
        h2mat.trialtransfermatrices,
        h2mat.couplingmatrices,
        h2mat.aggregationplan,
        h2mat.disaggregationplan,
        h2mat.scheduler,
        h2mat.dim,
    )
end

"""
    nearmatrix(h2mat::PetrovGalerkinNCA)

Return the near field of `h2mat` as a block-sparse matrix.
"""
function nearmatrix(h2mat::PetrovGalerkinNCA)
    return h2mat.nearinteractions
end

"""
    storage(h2mat::PetrovGalerkinNCA)

Print the memory footprint of `h2mat` and its compression ratio with respect to the dense
matrix.

Counts the entries of the near field, the coupling and transfer matrices, and the leaf
bases.

# Returns

  - Storage of these entries in GB.
"""
function storage(h2mat::PetrovGalerkinNCA)
    refsize = size(h2mat, 1) * size(h2mat, 2) * sizeof(eltype(h2mat))
    matsize = 0
    for blk in h2mat.nearinteractions.blocks
        matsize += length(blk)
    end
    for blk in h2mat.couplingmatrices.blocks
        matsize += length(blk)
    end
    for blk in h2mat.testtransfermatrices.blocks
        matsize += length(blk)
    end
    for blk in h2mat.trialtransfermatrices.blocks
        matsize += length(blk)
    end
    for blk in h2mat.nestedtestbases.blocks
        matsize += length(blk)
    end
    for blk in h2mat.nestedtrialbases.blocks
        matsize += length(blk)
    end
    println("storage: ", matsize * sizeof(eltype(h2mat)) * 10^-9, " GB")
    println("summary size: ", Base.summarysize(h2mat) * 10^-9, " GB")
    println("compression ratio: ", (matsize * sizeof(eltype(h2mat))) / refsize)
    return matsize * sizeof(eltype(h2mat)) * 10^-9
end

@views function LinearAlgebra.mul!(
    y::AbstractVector, A::PetrovGalerkinNCA, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(eltype(y)))

    xhat = zeros(eltype(y), A.aggregationplan.globalidxtocoeffptr[end] - 1)
    yhat = zeros(eltype(y), A.disaggregationplan.globalidxtocoeffptr[end] - 1)

    _project_to_coefficients!(
        xhat,
        x,
        A.nestedtrialbases,
        A.aggregationplan,
        node -> H2Trees.values(H2Trees.trialtree(A.tree), node),
        _idop,
        A.scheduler,
    )
    _aggregate_coefficients!(
        xhat, A.trialtransfermatrices, A.aggregationplan, _idop, A.scheduler
    )

    _couple_forward!(
        yhat,
        xhat,
        A.couplingmatrices,
        A.disaggregationplan,
        A.aggregationplan,
        _idop,
        A.scheduler,
    )

    _disaggregate_coefficients!(
        yhat, A.testtransfermatrices, A.disaggregationplan, _idop, A.scheduler
    )

    _project_to_output!(
        y,
        yhat,
        A.nestedtestbases,
        A.disaggregationplan,
        node -> H2Trees.values(H2Trees.testtree(A.tree), node),
        _idop,
        A.scheduler,
    )

    mul!(y, A.nearinteractions, x, true, true)

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVector,
    A::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(eltype(y)))

    xhat = zeros(eltype(y), A.lmap.disaggregationplan.globalidxtocoeffptr[end] - 1)
    yhat = zeros(eltype(y), A.lmap.aggregationplan.globalidxtocoeffptr[end] - 1)

    _project_to_coefficients!(
        xhat,
        x,
        A.lmap.nestedtestbases,
        A.lmap.disaggregationplan,
        node -> H2Trees.values(H2Trees.testtree(A.lmap.tree), node),
        transpose,
        A.lmap.scheduler,
    )
    _aggregate_coefficients!(
        xhat,
        A.lmap.testtransfermatrices,
        A.lmap.disaggregationplan,
        transpose,
        A.lmap.scheduler,
    )
    _couple_reverse!(
        yhat,
        xhat,
        A.lmap.couplingmatrices,
        A.lmap.aggregationplan,
        A.lmap.disaggregationplan,
        transpose,
        A.lmap.scheduler,
    )
    _disaggregate_coefficients!(
        yhat,
        A.lmap.trialtransfermatrices,
        A.lmap.aggregationplan,
        transpose,
        A.lmap.scheduler,
    )
    _project_to_output!(
        y,
        yhat,
        A.lmap.nestedtrialbases,
        A.lmap.aggregationplan,
        node -> H2Trees.values(H2Trees.trialtree(A.lmap.tree), node),
        transpose,
        A.lmap.scheduler,
    )

    mul!(y, transpose(A.lmap.nearinteractions), x, true, true)
    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVector,
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    LinearMaps.check_dim_mul(y, A, x)
    fill!(y, zero(eltype(y)))

    xhat = zeros(eltype(y), A.lmap.disaggregationplan.globalidxtocoeffptr[end] - 1)
    yhat = zeros(eltype(y), A.lmap.aggregationplan.globalidxtocoeffptr[end] - 1)

    _project_to_coefficients!(
        xhat,
        x,
        A.lmap.nestedtestbases,
        A.lmap.disaggregationplan,
        node -> H2Trees.values(H2Trees.testtree(A.lmap.tree), node),
        adjoint,
        A.lmap.scheduler,
    )
    _aggregate_coefficients!(
        xhat,
        A.lmap.testtransfermatrices,
        A.lmap.disaggregationplan,
        adjoint,
        A.lmap.scheduler,
    )
    _couple_reverse!(
        yhat,
        xhat,
        A.lmap.couplingmatrices,
        A.lmap.aggregationplan,
        A.lmap.disaggregationplan,
        adjoint,
        A.lmap.scheduler,
    )
    _disaggregate_coefficients!(
        yhat,
        A.lmap.trialtransfermatrices,
        A.lmap.aggregationplan,
        adjoint,
        A.lmap.scheduler,
    )
    _project_to_output!(
        y,
        yhat,
        A.lmap.nestedtrialbases,
        A.lmap.aggregationplan,
        node -> H2Trees.values(H2Trees.trialtree(A.lmap.tree), node),
        adjoint,
        A.lmap.scheduler,
    )

    mul!(y, adjoint(A.lmap.nearinteractions), x, true, true)
    return y
end
