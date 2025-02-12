#=using NestedCrossApproximation
using FastBEAST

struct Pivot
    loc::Vector{Int}
    glo::Vector{Int}
end

function pivotselection(
    tree::NminTree{D},
    fars::Vector{Vector{Tuple{Int,Int}}},
    pivstrat::PivStrat;
    maxrank=40,
    multithreading=true,
) where {D}
    pivots = Dict{Int,Vector{Int}}[]
    interactionlist = NestedCrossApproximation.sort_interactions(
        length(tree.nodes), fars; testortrial=1
    )
    clusterlink = FastBEAST.cluster_link(tree)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for (level, lcluster) in enumerate(clusterlink)
        if lcluster != []
            lpivots = Dict{Int,Vector{Int}}()
            _foreach(lcluster) do (cluster)
                cidcs = value(tree, interactionlist[cluster])
                parent = ClusterTrees.parent(tree, cluster)
                if interactionlist[parent] != []
                    append!(cidcs, pivots[end][parent])
                end
                if cidcs != []
                    refct = tree.nodes[cluster].node.data.ct
                    #local pivstrat
                    strat = pivstrat(cidcs, refct)
                    pivot = zeros(Int, maxrank)
                    for i in 1:min(maxrank, length(cidcs))
                        pivot[i] = strat()
                    end
                    push!(lpivots, cluster => cidcs[pivot])
                end
            end
            push!(pivots, lpivots)
        end
    end

    return pivots
end

function buildbasis(
    colbuffer::Union{Matrix{K},SubArray},
    pivotidcs::Union{SubArray,Vector{Int}},
    ridcs::Vector{Int},
    cidcs::Vector{Int},
) where {K}
    len = length(pivotidcs)
    @views basisblock = colbuffer[ridcs, 1:len] * colbuffer[pivotidcs, 1:len]^-1
    return H2BasisBlock(basisblock, ridcs, cidcs, Int[])
end

function buildtransfer(
    colbuffer::Union{Matrix{K},SubArray},
    rpivots::Dict{Int,Vector{Int}},
    children::Vector{Int},
    pivotidcs::Union{SubArray,Vector{Int}},
    ridcs::Vector{Int},
    cidcs::Vector{Int},
) where {K}
    transfer = Matrix{K}[]
    for child in children
        len = length(pivotidcs)
        push!(transfer, colbuffer[rpivots[child], 1:len] * inv(colbuffer[pivotidcs, 1:len]))
    end

    return H2BasisBlock(transfer, ridcs, cidcs, children)
end

function assembly(
    cpivots, tree, assembler, ::Type{K}; maxrank=40, tol=1e-4, multithreading=false
) where {K}
    #Buffer
    colbuffer = zeros(K, tree.num_elements, maxrank)
    rowbuffer = zeros(K, maxrank, tree.num_elements)
    rows = zeros(Int, tree.num_elements)

    moments = Dict{Int,H2BasisBlock{Int,K}}()
    transfer = Dict{Int,H2BasisBlock{Int,K}}()
    rpivots = Dict{Int,Vector{Int}}()
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    for level in reverse(cpivots)
        _foreach(level) do (nodeid, pivots)
            ridcs = value(tree, nodeid)
            lm = FastBEAST.LazyMatrix(assembler, ridcs, pivots, K)
            @views loc = NestedCrossApproximation.pca(
                lm,
                rows[ridcs],
                rowbuffer[1:maxrank, ridcs],
                colbuffer[ridcs, 1:maxrank],
                MaximumValue();
                tol=tol,
            )
            push!(rpivots, nodeid => ridcs[loc])

            if ClusterTrees.haschildren(tree, nodeid)
                chds = Int[]
                for child in children(tree, nodeid)
                    push!(chds, child)
                end
                @views push!(
                    transfer,
                    nodeid =>
                        buildtransfer(colbuffer, rpivots, chds, ridcs[loc], ridcs, ridcs),
                )
            else
                @views push!(
                    moments, nodeid => buildbasis(colbuffer, ridcs[loc], ridcs, ridcs)
                )
            end
        end
    end

    return moments, transfer, rpivots
end
=#
