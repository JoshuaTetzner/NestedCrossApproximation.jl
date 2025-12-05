struct PetrovGalerkinNCA{
    T,TreeType,NearInteractionType,NestedBasesDict,TransferMatrixType,CouplingMatrixType
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    nearinteractions::NearInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    dim::Tuple{Int,Int}
    ntasks::Int

    function PetrovGalerkinNCA{T}(
        tree,
        nearinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        dim,
        ntasks,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(nearinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
        }(
            tree,
            nearinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            dim,
            ntasks,
        )
    end
end

function defaultfarquadstrat(operator, testspace, trialspace) end

function defaultnearquadstrat(operator, testspace, trialspace) end

function PetrovGalerkinNCA(
    operator,
    testspace,
    trialspace,
    tree;
    tol=1e-4,
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
    ntasks=Threads.nthreads(),
    isnear=H2Trees.isnear,
    maxrank=40,
)

    # near interactions
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    values, nearvalues = H2Trees.nearinteractions(
        tree; isnear=isnear, extractselfvalues=false
    )
    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    @tasks for i in eachindex(values)
        @set ntasks = ntasks
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nearinteractions = BlockSparseMatrix(
        blocks,
        values,
        nearvalues,
        size(nearmatrix);
        scheduler=DynamicScheduler(; ntasks=ntasks),
    )

    # far interactions
    #testdfars, trialfars = farinteractions(tree; isnear=isnear)

    farmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=farquadstrat
    )

    println("compress_testtree")
    nestedtestbases, testtransfermatrices, testpivots = testcompressor(
        farmatrix,
        #testdfars,
        tree,
        reverse(testbuffer(testcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks));
        isnear=isnear,
        ntasks=ntasks,
    )

    println("compress_trialtree")
    nestedtrialbases, trialtransfermatrices, trialpivots = trialcompressor(
        farmatrix,
        #trialfars,
        tree,
        trialbuffer(trialcompressor, farmatrix; maxrank=maxrank, ntasks=ntasks);
        isnear=isnear,
        ntasks=ntasks,
    )

    println("coupling")
    couplingmatrices = assemble_couplingmatrices(
        farmatrix, tree, testpivots, trialpivots; isnear=isnear, ntasks=ntasks
    )

    return PetrovGalerkinNCA{eltype(nearmatrix)}(
        tree,
        nearinteractions,
        Dict(
            i => nestedtestbases[i] for
            i in eachindex(nestedtestbases) if isassigned(nestedtestbases, i)
        ),
        Dict(
            i => nestedtrialbases[i] for
            i in eachindex(nestedtrialbases) if isassigned(nestedtrialbases, i)
        ),
        Dict(
            i => testtransfermatrices[i] for
            i in eachindex(testtransfermatrices) if isassigned(testtransfermatrices, i)
        ),
        Dict(
            i => trialtransfermatrices[i] for
            i in eachindex(trialtransfermatrices) if isassigned(trialtransfermatrices, i)
        ),
        Dict(
            i => couplingmatrices[i] for
            i in eachindex(couplingmatrices) if isassigned(couplingmatrices, i)
        ),
        size(farmatrix),
        ntasks,
    )
end
