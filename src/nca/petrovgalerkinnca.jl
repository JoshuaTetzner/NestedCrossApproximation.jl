"""
    PetrovGalerkinNCACore{T}

Read-only part of an H²-matrix constructed with the nested cross approximation.

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
  - `scheduler`: default scheduler of a workspace, created from `ntasks` by the constructor
  - `dim::Tuple{Int,Int}`: matrix dimensions `(m, n)`

# Notes

A core is not a linear map. Wrap it in an [`H2Workspace`](@ref), which owns the buffers of
the matrix-vector product and is the `LinearMap`:

```julia
core = PetrovGalerkinNCACore(operator, testspace, trialspace, tree)
h2mat = H2Workspace(core)
mul!(y, h2mat, x)
```

A matrix-vector product does not write to the core, so several workspaces may share one
core and run concurrently. [`PetrovGalerkinNCA`](@ref) builds a core and one workspace
in a single call.

# See also

[`H2Workspace`](@ref), [`PetrovGalerkinNCA`](@ref)
"""
struct PetrovGalerkinNCACore{
    T,
    TreeType,
    NearInteractionType,
    NestedBasesDict,
    TransferMatrixType,
    CouplingMatrixType,
    PlanType,
    SchedulerType,
}
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

    function PetrovGalerkinNCACore{T}(
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
    PetrovGalerkinNCACore(operator, testspace, trialspace, tree; kwargs...)

Assemble the read-only part of an H²-matrix of `operator` with the nested cross approximation.

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

  - [`PetrovGalerkinNCACore`](@ref)
"""
function PetrovGalerkinNCACore(
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
    return PetrovGalerkinNCACore{eltype(farmatrix)}(
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

Base.size(A::PetrovGalerkinNCACore) = A.dim
Base.size(A::PetrovGalerkinNCACore, dim::Int) = A.dim[dim]
Base.eltype(::PetrovGalerkinNCACore{T}) where {T} = T

"""
    ConcurrentWorkspaceError

Thrown when a matrix-vector product starts on an [`H2Workspace`](@ref) that is in use by
another product.
"""
struct ConcurrentWorkspaceError <: Exception end

function Base.showerror(io::IO, ::ConcurrentWorkspaceError)
    return print(
        io,
        "ConcurrentWorkspaceError: the workspace is in use by another matrix-vector ",
        "product. Create one H2Workspace per task; they can share the core.",
    )
end

"""
    H2Workspace(core; scheduler=core.scheduler)
    H2Workspace(core, T; scheduler=core.scheduler)

H²-matrix as a linear map: a read-only `core` and the buffers of the matrix-vector product
for vectors of element type `T` (default `eltype(core)`).

# Fields

  - `core`: read-only part, e.g. a [`PetrovGalerkinNCACore`](@ref)
  - `trialcoeffs`, `testcoeffs`: coefficients of the trial and the test clusters
  - `scheduler`: `OhMyThreads` scheduler of the matrix-vector product
  - `busy`: set during a matrix-vector product

# Notes

A workspace supports `*`, `mul!`, `transpose`, and `adjoint`. The fields of the core are
also available as properties of the workspace, e.g. `h2mat.nearinteractions`.

One workspace runs one matrix-vector product at a time: a second product on the same
workspace throws a [`ConcurrentWorkspaceError`](@ref). To multiply with the same matrix
from several tasks, give each task its own workspace over the shared core:

```julia
workspaces = [H2Workspace(core; scheduler=OhMyThreads.SerialScheduler()) for _ in 1:4]
```

For vectors of an element type other than `T` the buffers are allocated in every product.

# See also

[`core`](@ref), [`PetrovGalerkinNCACore`](@ref), [`PetrovGalerkinNCA`](@ref)
"""
struct H2Workspace{T,CoreType,BufferType,SchedulerType} <: LinearMaps.LinearMap{T}
    core::CoreType
    trialcoeffs::Vector{BufferType}
    testcoeffs::Vector{BufferType}
    scheduler::SchedulerType
    busy::Threads.Atomic{Bool}
end

function H2Workspace(
    core::PetrovGalerkinNCACore, ::Type{T}=eltype(core); scheduler=core.scheduler
) where {T}
    return H2Workspace{eltype(core),typeof(core),T,typeof(scheduler)}(
        core,
        zeros(T, core.aggregationplan.globalidxtocoeffptr[end] - 1),
        zeros(T, core.disaggregationplan.globalidxtocoeffptr[end] - 1),
        scheduler,
        Threads.Atomic{Bool}(false),
    )
end

"""
    core(h2mat::H2Workspace)

Return the read-only part of `h2mat`, which can be shared with further workspaces.
"""
core(h2mat::H2Workspace) = getfield(h2mat, :core)
core(A::LinearMaps.TransposeMap{<:Any,<:H2Workspace}) = core(A.lmap)
core(A::LinearMaps.AdjointMap{<:Any,<:H2Workspace}) = core(A.lmap)

function Base.getproperty(h2mat::H2Workspace, name::Symbol)
    hasfield(H2Workspace, name) && return getfield(h2mat, name)
    return getproperty(getfield(h2mat, :core), name)
end

function Base.propertynames(h2mat::H2Workspace)
    return (fieldnames(H2Workspace)..., propertynames(core(h2mat))...)
end

# Run `f` as the only matrix-vector product on `h2mat`. Fails instead of waiting: a
# workspace shared between tasks is a bug of the caller.
function withworkspace(f, h2mat::H2Workspace)
    Threads.atomic_cas!(h2mat.busy, false, true) && throw(ConcurrentWorkspaceError())
    try
        return f()
    finally
        h2mat.busy[] = false
    end
end

# Zeroed coefficient vector: the buffer of the workspace if its element type fits.
_coefficients(buffer::Vector{T}, ::Type{T}) where {T} = fill!(buffer, zero(T))
_coefficients(buffer::Vector, ::Type{T}) where {T} = zeros(T, length(buffer))

"""
    PetrovGalerkinNCA{T}

[`H2Workspace`](@ref) over a [`PetrovGalerkinNCACore`](@ref): an H²-matrix constructed with
the nested cross approximation, as a linear map.
"""
const PetrovGalerkinNCA{T} = H2Workspace{T,<:PetrovGalerkinNCACore}

"""
    PetrovGalerkinNCA(operator, testspace, trialspace, tree; kwargs...)

Assemble an H²-matrix of `operator` with the nested cross approximation and return it as a
linear map: a [`PetrovGalerkinNCACore`](@ref) and one [`H2Workspace`](@ref) over it.

The keyword arguments are those of [`PetrovGalerkinNCACore`](@ref).

# Notes

The returned workspace runs one matrix-vector product at a time. For concurrent products
build further workspaces over `core(h2mat)`, see [`H2Workspace`](@ref).
"""
function PetrovGalerkinNCA(operator, testspace, trialspace, tree; kwargs...)
    return H2Workspace(
        PetrovGalerkinNCACore(operator, testspace, trialspace, tree; kwargs...)
    )
end

Base.size(A::PetrovGalerkinNCA) = A.dim
Base.size(A::PetrovGalerkinNCA, dim::Int) = A.dim[dim]

"""
    farmatrix(h2mat::PetrovGalerkinNCA)

Return the far field of `h2mat` as a [`PetrovGalerkinNCA`](@ref) with an empty near field.

The nested bases, transfer and coupling matrices are shared with `h2mat`, the buffers of
the matrix-vector product are not.
"""
function farmatrix(h2mat::PetrovGalerkinNCA)
    blocks = Matrix{eltype(h2mat)}[]
    nears = BlockSparseMatrix(blocks, Vector{Int}[], Vector{Int}[], h2mat.dim)

    farcore = PetrovGalerkinNCACore{eltype(h2mat)}(
        h2mat.tree,
        nears,
        h2mat.nestedtestbases,
        h2mat.nestedtrialbases,
        h2mat.testtransfermatrices,
        h2mat.trialtransfermatrices,
        h2mat.couplingmatrices,
        h2mat.aggregationplan,
        h2mat.disaggregationplan,
        core(h2mat).scheduler,
        h2mat.dim,
    )
    return H2Workspace(farcore, eltype(h2mat.trialcoeffs); scheduler=h2mat.scheduler)
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

# `_unsafe_mul!` and not `mul!`: this is what `LinearMaps` calls for a map inside a sum or
# a product of maps, where a method of `mul!` alone is bypassed.
function LinearMaps._unsafe_mul!(
    y::AbstractVector,
    A::Union{
        PetrovGalerkinNCA,
        LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinNCA},
        LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinNCA},
    },
    x::AbstractVector,
)
    return withworkspace(() -> _mul!(y, A, x), _workspace(A))
end

_workspace(A::PetrovGalerkinNCA) = A
_workspace(A::LinearMaps.TransposeMap) = A.lmap
_workspace(A::LinearMaps.AdjointMap) = A.lmap

@views function _mul!(y::AbstractVector, A::PetrovGalerkinNCA, x::AbstractVector)
    fill!(y, zero(eltype(y)))

    xhat = _coefficients(A.trialcoeffs, eltype(y))
    yhat = _coefficients(A.testcoeffs, eltype(y))

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

@views function _mul!(
    y::AbstractVector,
    A::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    fill!(y, zero(eltype(y)))

    xhat = _coefficients(A.lmap.testcoeffs, eltype(y))
    yhat = _coefficients(A.lmap.trialcoeffs, eltype(y))

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

@views function _mul!(
    y::AbstractVector,
    A::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinNCA},
    x::AbstractVector,
)
    fill!(y, zero(eltype(y)))

    xhat = _coefficients(A.lmap.testcoeffs, eltype(y))
    yhat = _coefficients(A.lmap.trialcoeffs, eltype(y))

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
