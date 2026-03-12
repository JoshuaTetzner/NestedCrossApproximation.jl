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
    matrixdata=defaultmatrixdata(operator, testspace, trialspace),
    farmatrixdata=defaultfarmatrixdata(operator, testspace, trialspace),
    testcompressor=TopDown(),
    trialcompressor=TopDown(),
    scheduler=DynamicScheduler(),
    isnear=isnear(),
    maxrank=40,
)
    @time nears = assemblenears(
        operator,
        testspace,
        trialspace,
        tree;
        isnear=isnear,
        scheduler=scheduler,
        matrixdata=matrixdata,
    )
    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; matrixdata=farmatrixdata
    )
    #testfarptr, trialfarptr, testfars, trialfars = farinteractions(tree; isnear=isnear)
    testfardata, trialfardata = fardata(tree, isnear)
    testbuf = testbuffer(
        testcompressor, farmatrix; maxrank=maxrank, ntasks=Threads.nthreads()
    )
    trialbuf = trialbuffer(
        trialcompressor, farmatrix; maxrank=maxrank, ntasks=Threads.nthreads()
    )
    #=
    @time begin
        nestedtestbases, testtransfermats, testpivots = testcompressor(
            farmatrix,
            tree,
            testfarptr,
            testfars,
            reverse(testbuf);
            scheduler=scheduler,
            maxrank=maxrank,
        )

        nestedtrialbases, trialtransfermats, trialpivots = trialcompressor(
            farmatrix,
            trialfarptr,
            trialfars,
            tree,
            trialbuf;
            scheduler=scheduler,
            maxrank=maxrank,
        )

        couplingmatrices = assemble_couplingstore(
            farmatrix, testpivots, trialpivots, testfars, testfarptr; scheduler=scheduler
        )

        aggregationplan = plan_from_pivots(trialpivots)
        disaggregationplan = plan_from_pivots(testpivots)
    end=#
    @time begin
        nestedtestbases, testtransfermats, testpivots = testcompressor(
            farmatrix,
            tree,
            testfardata,
            reverse(testbuf);
            scheduler=scheduler,
            maxrank=maxrank,
        )

        nestedtrialbases, trialtransfermats, trialpivots = trialcompressor(
            farmatrix, tree, trialfardata, trialbuf; scheduler=scheduler, maxrank=maxrank
        )

        couplingmatrices = assemble_couplingstore(
            farmatrix, testpivots, trialpivots, testfardata; scheduler=scheduler
        )

        aggregationplan = plan_from_pivots(trialpivots)
        disaggregationplan = plan_from_pivots(testpivots)
    end

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
        scheduler,
        (length(testspace), length(trialspace)),
    )
end

Base.size(A::PetrovGalerkinNCA) = A.dim
Base.size(A::PetrovGalerkinNCA, dim::Int) = A.dim[dim]
Base.eltype(::PetrovGalerkinNCA{T}) where {T} = T

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
    LinearMaps.check_dim_mul(y, A.lmap, x)
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
    LinearMaps.check_dim_mul(y, A.lmap, x)
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
