using LinearMaps

struct PetrovGalerkinWNCA{
    T,
    TreeType,
    DirectionTreeType,
    NearInteractionType,
    LFInteractionType,
    NestedBasesDict,
    TransferMatrixType,
    CouplingMatrixType,
} <: LinearMaps.LinearMap{T}
    tree::TreeType
    dtree::DirectionTreeType
    nearinteractions::NearInteractionType
    lowfrequencyinteractions::LFInteractionType
    nestedtestbases::NestedBasesDict
    nestedtrialbases::NestedBasesDict
    testtransfermatrices::TransferMatrixType
    trialtransfermatrices::TransferMatrixType
    couplingmatrices::CouplingMatrixType
    dim::Tuple{Int,Int}
    ismultithreaded::Bool

    function PetrovGalerkinWNCA{T}(
        tree,
        dtree,
        nearinteractions,
        lowfrequencyinteractions,
        nestedtestbases,
        nestedtrialbases,
        testtransfermatrices,
        trialtransfermatrices,
        couplingmatrices,
        dim,
        ismultithreaded,
    ) where {T}
        return new{
            T,
            typeof(tree),
            typeof(dtree),
            typeof(nearinteractions),
            typeof(lowfrequencyinteractions),
            typeof(nestedtestbases),
            typeof(testtransfermatrices),
            typeof(couplingmatrices),
        }(
            tree,
            dtree,
            nearinteractions,
            lowfrequencyinteractions,
            nestedtestbases,
            nestedtrialbases,
            testtransfermatrices,
            trialtransfermatrices,
            couplingmatrices,
            dim,
            ismultithreaded,
        )
    end
end

function isnear(k, treea, treeb, nodea, nodeb; ηₗ=1.0, ηₕ=4.0)
    ths = H2Trees.halfsize(treea, nodea) * sqrt(3)
    shs = H2Trees.halfsize(treeb, nodeb) * sqrt(3)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    if k / pi * 4 * min(ths, shs) <= 1
        (2 * max(ths, shs) <= ηₗ * max(dist, 0.0)) ? (return false) : (return true)
    else
        (4 * k * max(ths^2, shs^2) <= ηₕ * max(dist, 0.0)) ? (return false) : (return true)
    end
end

function islf(k, tree, level)
    return k / pi * 4 * sqrt(3) * H2Trees.halfsize(tree) / 2^(level - 1) <= 1
end

function islf(k)
    lf(tree, level) = islf(k, tree, level)
    return lf
end

function PetrovGalerkinWNCA(
    operator,
    testspace,
    trialspace,
    tree;
    farquadstrat=defaultfarquadstrat(operator, testspace, trialspace),
    nearquadstrat=defaultnearquadstrat(operator, testspace, trialspace),
    testcompressor=TopDownCompressor(),
    trialcompressor=TopDownCompressor(),
    lfcompressor=AdaptiveCrossApproximation.ACA(),
    ntasks=Threads.nthreads(),
    isnear=H2Trees.isnear,
    islf=islf(wavenumber(operator)),
    maxrank=40,
)
    # near interactions
    nearmatrix = AbstractKernelMatrix(
        operator, testspace, trialspace; quadstrat=nearquadstrat
    )
    values, nearvalues, fars, dirs, lfvalues, lffarvalues = directionalitneractions(
        tree, islf, isnear
    )
    blocks = Vector{Matrix{eltype(nearmatrix)}}(undef, length(values))
    println("nears")
    Threads.@threads for i in eachindex(values)
        blk = zeros(eltype(nearmatrix), length(values[i]), length(nearvalues[i]))
        nearmatrix(blk, values[i], nearvalues[i])
        blocks[i] = blk
    end
    nearinteractions = BlockSparseMatrix(blocks, values, nearvalues, size(nearmatrix))

    println("lfs")
    lk = Threads.SpinLock()
    rowbuffer = (maxrank, maximum(length.(Iterators.flatten(lffarvalues))))
    colbuffer = (maximum(length.(lfvalues)))
    lfblocks = MatrixBlock{Int,eltype(farmatrix),LowRankMatrix{eltype(farmatrix)}}[]
    am = allocate_lfbuffer(())
    for (levelidx, level) in enumerate(lfvalues)
        @tasks for (tidx, t) in enumerate(lfvalues[level])
            @set ntasks = ntasks
            for s in lffarvalues[levelidx][tidx]
                compress()
                blk = MatrixBlock(LowRankMatrix(U, V), t, s)
                lock(lk) do
                    push!(blk, lfblocks)
                end
            end
        end
    end

    println("hf")
    nlev = 0
    for f in hffars
        if f != []
            nlev += 1
        end
    end
    tol = tol / nlev
    @time tbases, tpivots, tdirfars, Ftpivots, sbases, spivots, sdirfars, Fspivots, dtree = directionalcompressor(
        farassembler,
        testtree,
        hffars,
        imag(operator.gamma);
        tlrf=testcompressor,
        slrf=trialcompressor,
        maxrank=maxrank,
        tol=tol,
        ηₕ=ηₕ,
        multithreading=multithreading,
    )

    @time testbases, testtransfer, trialbases, trialtransfer = builddirectionalH2(
        testtree, dtree, hffars, tbases, tpivots, sbases, spivots; multithreading=true
    )

    @time coupling = computecoupling(
        farassembler, tpivots, tdirfars, spivots, sdirfars, reduce(vcat, hffars)
    )

    return PetrovGalerkinWNCA{ComplexF64}(
        blktree,
        dtree,
        nearinteractions,
        lfinteractions,
        testbases,
        trialbases,
        testtransfer,
        trialtransfer,
        coupling,
        (testtree.num_elements, trialtree.num_elements),
        multithreading,
    )
end

function Base.size(A::PetrovGalerkinWNCA, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    else
        error("dim must be either 1 or 2")
    end
end

function Base.size(A::Adjoint{T}, dim=nothing) where {T<:PetrovGalerkinWNCA}
    if dim === nothing
        return reverse(A.dim[1], A.dim[2])
    elseif dim == 1
        return h2mat.lmap.dim[2]
    elseif dim == 2
        return h2mat.lmap.dim[1]
    else
        error("dim must be either 1 or 2")
    end
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, A::PetrovGalerkinWNCA, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, moment) in A.nestedtrialbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, dir * x[value(A.tree.trial_cluster, idx)])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.trialtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                childs = collect(ClusterTrees.children(A.tree.trial_cluster, node))
                xhatdir = transfers[1] * xhat[childs[1]][ClusterTrees.parent(A.dtree, dir)]
                for (i, transfer) in enumerate(transfers[2:end])
                    xhatdir +=
                        transfer * xhat[childs[i + 1]][ClusterTrees.parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.row_basis)
            if haskey(yhat[lrb.row_basis], lrb.row_dir)
                yhat[lrb.row_basis][lrb.row_dir] += lrb.Z * xhat[lrb.col_basis][lrb.col_dir]
            else
                yhat[lrb.row_basis][lrb.row_dir] = lrb.Z * xhat[lrb.col_basis][lrb.col_dir]
            end
        else
            yhat[lrb.row_basis] = Dict(
                lrb.row_dir => lrb.Z * xhat[lrb.col_basis][lrb.col_dir]
            )
        end
    end

    for nodes in A.testtransfermatrices
        for (node, data) in nodes
            childs = collect(ClusterTrees.children(A.tree.test_cluster, node))

            for (dir, transfers) in data
                childdir = ClusterTrees.parent(A.dtree, dir)
                for (i, child) in enumerate(childs)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] += transfers[i] * yhat[node][dir]
                        else
                            yhat[child][childdir] = transfers[i] * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(childdir => transfers[i] * yhat[node][dir])
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtestbases
        for (dir, basis) in moment
            y[value(A.tree.test_cluster, idx)] += basis * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
    end

    y += A.nearinteractions * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.TransposeMap{<:Any,<:PetrovGalerkinWNCA},
    x::AbstractVector,
)
    A = At.lmap
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, moment) in A.nestedtestbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, transpose(dir) * x[value(A.tree.test_cluster, idx)])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.testtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                childs = collect(ClusterTrees.children(A.tree.test_cluster, node))
                xhatdir =
                    transpose(transfers[1]) *
                    xhat[childs[1]][ClusterTrees.parent(A.dtree, dir)]
                for (i, transfer) in enumerate(transfers[2:end])
                    xhatdir +=
                        transpose(transfer) *
                        xhat[childs[i + 1]][ClusterTrees.parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            if haskey(yhat[lrb.col_basis], lrb.col_dir)
                yhat[lrb.col_basis][lrb.col_dir] +=
                    transpose(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            else
                yhat[lrb.col_basis][lrb.col_dir] =
                    transpose(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            end
        else
            yhat[lrb.col_basis] = Dict(
                lrb.col_dir => transpose(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            )
        end
    end

    for nodes in A.trialtransfermatrices
        for (node, data) in nodes
            childs = collect(ClusterTrees.children(A.tree.trial_cluster, node))
            for (dir, transfers) in data
                childdir = ClusterTrees.parent(A.dtree, dir)
                for (i, child) in enumerate(childs)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] +=
                                transpose(transfers[i]) * yhat[node][dir]
                        else
                            yhat[child][childdir] =
                                transpose(transfers[i]) * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(
                            childdir => transpose(transfers[i]) * yhat[node][dir]
                        )
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtrialbases
        for (dir, basis) in moment
            y[value(A.tree.trial_cluster, idx)] += transpose(basis) * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.σ] += transpose(lrb.M) * x[lrb.τ]
    end

    y += transpose(A.nearinteractions) * x

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    At::LinearMaps.AdjointMap{<:Any,<:PetrovGalerkinWNCA},
    x::AbstractVector,
)
    A = At.lmap
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))

    xhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.trial_cluster.nodes))
    yhat = Vector{Dict{Int,Vector{eltype(y)}}}(undef, length(A.tree.test_cluster.nodes))

    for (idx, moment) in A.nestedtestbases
        res = Vector{eltype(y)}[]
        for (k, dir) in moment
            push!(res, adjoint(dir) * x[value(A.tree.test_cluster, idx)])
        end
        xhat[idx] = Dict(keys(moment) .=> res)
    end

    for nodes in reverse(A.testtransfermatrices)
        for (node, data) in nodes
            res = Vector{eltype(y)}[]
            for (dir, transfers) in data
                childs = collect(ClusterTrees.children(A.tree.test_cluster, node))
                xhatdir =
                    adjoint(transfers[1]) *
                    xhat[childs[1]][ClusterTrees.parent(A.dtree, dir)]
                for (i, transfer) in enumerate(transfers[2:end])
                    xhatdir +=
                        adjoint(transfer) *
                        xhat[childs[i + 1]][ClusterTrees.parent(A.dtree, dir)]
                end
                push!(res, xhatdir)
            end
            xhat[node] = Dict(keys(data) .=> res)
        end
    end

    for lrb in A.couplingmatrices
        if isassigned(yhat, lrb.col_basis)
            if haskey(yhat[lrb.col_basis], lrb.col_dir)
                yhat[lrb.col_basis][lrb.col_dir] +=
                    adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            else
                yhat[lrb.col_basis][lrb.col_dir] =
                    adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            end
        else
            yhat[lrb.col_basis] = Dict(
                lrb.col_dir => adjoint(lrb.Z) * xhat[lrb.row_basis][lrb.row_dir]
            )
        end
    end

    for nodes in A.trialtransfermatrices
        for (node, data) in nodes
            childs = collect(ClusterTrees.children(A.tree.trial_cluster, node))
            for (dir, transfers) in data
                childdir = ClusterTrees.parent(A.dtree, dir)
                for (i, child) in enumerate(childs)
                    if isassigned(yhat, child)
                        if haskey(yhat[child], childdir)
                            yhat[child][childdir] += adjoint(transfers[i]) * yhat[node][dir]
                        else
                            yhat[child][childdir] = adjoint(transfers[i]) * yhat[node][dir]
                        end
                    else
                        yhat[child] = Dict(
                            childdir => adjoint(transfers[i]) * yhat[node][dir]
                        )
                    end
                end
            end
        end
    end

    for (idx, moment) in A.nestedtrialbases
        for (dir, basis) in moment
            y[value(A.tree.trial_cluster, idx)] += adjoint(basis) * yhat[idx][dir]
        end
    end

    for lrb in A.lowfrequencyinteractions
        y[lrb.σ] += adjoint(lrb.M) * x[lrb.τ]
    end

    y += adjoint(A.nearinteractions) * x

    return y
end
