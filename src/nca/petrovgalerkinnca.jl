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
    scheduler=DynamicScheduler(),
    isnear=isnear(),
    maxrank=40,
)
    println("nearinteractions")
    #=@time nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        scheduler=scheduler,
        matrixdata=matrixdata,
    )=#
    farmatrix = AdaptiveCrossApproximation.AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=farmatrixdata
    )
    println("fardata")
    @time testfardata, trialfardata = fardata(tree, isnear)

    println("testcompressor")
    @time nestedtestbases, testtransfermats, testpivots = testbases(
        testcompressor, farmatrix, tree, testfardata; scheduler=scheduler, maxrank=maxrank
    )
    println("trialcompressor")
    @time nestedtrialbases, trialtransfermats, trialpivots = trialbases(
        trialcompressor, farmatrix, tree, trialfardata; scheduler=scheduler, maxrank=maxrank
    )

    println("couplingmatrices")
    @time couplingmatrices = assemble_couplingstore(
        farmatrix, testpivots, trialpivots, testfardata, trialfardata; scheduler=scheduler
    )
    aggregationplan = plan_from_pivots(trialpivots)
    disaggregationplan = plan_from_pivots(testpivots)

    return PetrovGalerkinNCA{eltype(farmatrix)}(
        tree,
        nears,
        nestedtestbases,
        nestedtrialbases,
        testtransfermats,
        trialtransfermats,
        couplingmatrices,
        aggregationplan,
        disaggregationplan,
        SerialScheduler(),
        #scheduler,
        (length(testspace), length(trialspace)),
    )
end

Base.size(A::PetrovGalerkinNCA) = A.dim
Base.size(A::PetrovGalerkinNCA, dim::Int) = A.dim[dim]
Base.eltype(::PetrovGalerkinNCA{T}) where {T} = T

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

function nearmatrix(h2mat::PetrovGalerkinNCA)
    return h2mat.nearinteractions
end

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

    xhat = zeros(eltype(y), A.aggregationplan.ptr[end] - 1)
    yhat = zeros(eltype(y), A.disaggregationplan.ptr[end] - 1)

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

    xhat = zeros(eltype(y), A.lmap.disaggregationplan.ptr[end] - 1)
    yhat = zeros(eltype(y), A.lmap.aggregationplan.ptr[end] - 1)

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

    xhat = zeros(eltype(y), A.lmap.disaggregationplan.ptr[end] - 1)
    yhat = zeros(eltype(y), A.lmap.aggregationplan.ptr[end] - 1)

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
