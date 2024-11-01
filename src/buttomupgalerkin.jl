using NestedCrossApproximation
using FastBEAST

struct Pivot
    loc::Vector{Int}
    glo::Vector{Int}
end

function pivotselection(
    tree::NminTree{D}, fars::Vector{Vector{Tuple{Int, Int}}}, pivstrat::PivStrat;
    maxrank=40, multithreading=true
) where D

    pivots = Dict{Int, Vector{Int}}[] 
    interactionlist = NestedCrossApproximation.sort_interactions(
       length(tree.nodes), fars; testortrial=1
    )   
    clusterlink = FastBEAST.cluster_link(tree)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    
    #every cluster with parent interactions needs pivots
    for lcluster in clusterlink
        if lcluster != []
            lpivots = Dict{Int, Vector{Int}}()
            _foreach(lcluster) do (cluster) 
                cidcs = value(tree, interactionlist[cluster])
                parent = ClusterTrees.parent(tree, cluster)
                if interactionlist[parent] != []
                    append!(cidcs, pivots[parent].glo)
                end
                if cidcs != []
                    refct = tree.nodes[cluster].node.data.ct
                    #local pivstrat
                    strat = pivstrat(cidcs, refct)
                    pivot = zeros(Int, maxrank)
                    for i in 1:min(maxrank, length(cidcs))
                        pivot[i] = strat()
                    end
                    push!(lpivots, cluster =>cidcs[pivot])
                end
            end
            push!(pivots, lpivots)
        end
    end

    return pivots
end

function createbases(
    tree::NminTree{D}, colbuffer::Matrix{K}, rpivots::Vector{Pivot}, cpivots::Vector{Pivot};
    multithreading=true
) where {D, K}
    basis = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(eachindex(rpivots)) do leaf
        if isassigned(rpivots, leaf)
            len = length(rpivots[leaf].loc)
            @views basisblock = colbuffer[rpivots[leaf].glo, 1:len] * 
                colbuffer[rpivots[leaf].glo[rpivots[leaf].loc], 1:len]^-1
            basis[leaf] = H2BasisBlock(
                basisblock,
                #global-selected-pivots
                rpivots[leaf].glo,#[rpivots[leaf].loc],
                #local-selected-pivots
                cpivots[leaf].glo,
                #children
                Int[],
            )
        end
    end

    return basis
end

function createtransfer(
    tree, id::Int, colbuffer::Matrix{K}, rpivots::Vector{Pivot}, cpivots::Vector{Pivot};
     multithreading=true
) where K

    transfer = Matrix{K}[]
    children = Int[]
    @views τ = rpivots[id].glo[rpivots[id].loc]
    for child in ClusterTrees.children(tree, id)
        @views τc = rpivots[child].glo[rpivots[child].loc]
        @views push!(transfer, colbuffer[τc, 1:length(τ)] * colbuffer[τ, 1:length(τ)]^-1)
        push!(children, child)
    end

    return transfer, children
end

function assembly(
    cpivots, tree, assembler, ::Type{K}; maxrank=40, tol=1e-4, multithreading=false
) where K

    #Buffer
    colbuffer = zeros(K, tree.num_elements, maxrank)
    rowbuffer = zeros(K, maxrank, tree.num_elements)
    rows = zeros(Int, tree.num_elements)


    rpivots = Vector{Pivot}(undef, length(tree.nodes))
    #clusterlink = FastBEAST.cluster_link(tree)
    #leaves = [leaf for leaf in ClusterTrees.leaves(tree)]
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(leaves) do (key)
        if isassigned(cpivots, key)
            lm = FastBEAST.LazyMatrix(assembler, value(tree, key), cpivots[key].glo, K)
            @views loc = NestedCrossApproximation.pca(
                lm,
                rows[value(tree, key)],
                #cols[value(tree, key)],
                rowbuffer[1:maxrank, value(tree, key)],
                colbuffer[value(tree, key), 1:maxrank],
                #Vector(1:40),
                MaximumValue(),
                tol=tol
            )
            rpivots[key] = Pivot(loc, value(tree, key))
        end
    end
    moments = createbases(tree, colbuffer, rpivots, cpivots, multithreading=multithreading)
    transfer = Vector{H2BasisBlock{Int,K}}(undef, length(tree.nodes))
    for level in reverse(clusterlink[1:end-1])
        _foreach(level) do (key)
            if isassigned(cpivots, key) && !isassigned(moments, key)
                #reduced
                rowss = Int[]
                for child in ClusterTrees.children(tree, key)
                    append!(rowss, rpivots[child].glo[rpivots[child].loc])
                end 
                lm = FastBEAST.LazyMatrix(assembler, rowss, cpivots[key].glo, K)
                @views loc = NestedCrossApproximation.pca(
                    lm,
                    rows[rowss],
                    rowbuffer[1:(maxrank), rowss],
                    colbuffer[rowss, 1:(maxrank)],
                    #Vector(1:40),
                    MaximumValue(),
                    tol=tol
                )
                rpivots[key] = Pivot(loc, rowss)
                #full
                #=
                lm = FastBEAST.LazyMatrix(assembler, value(tree, key), cpivots[key].glo, K)
                @views loc = NestedCrossApproximation.pca(
                    lm,
                    rows[value(tree, key)],
                    rowbuffer[1:(maxrank), value(tree, key)],
                    colbuffer[value(tree, key), 1:maxrank],
                    #Vector(1:40),
                    MaximumValue(),
                    tol=tol
                )

                rpivots[key] = Pivot(loc, value(tree, key))
                =#
                
                T, children = createtransfer(tree, key, colbuffer, rpivots, cpivots)
                transfer[key] = H2BasisBlock(
                    T,
                    #global-selected-pivots
                    rpivots[key].glo[rpivots[key].loc],
                    #local-selected-pivots
                    cpivots[key].loc,
                    #children
                    children,
                )
            end
        end
    end

    return moments, transfer, rpivots
end

function assemble_couplingmatrices(
    matrixassembler::Function,
    ::Type{K},
    fars::Vector{Vector{Tuple{I,I}}},
    rpivots::Vector{Pivot},
    cpivots::Vector{Pivot};
    multithreading=true, 
) where {I, K}
    fars = reduce(vcat, fars)
    lowrankblocks = Vector{H2MatrixBlock{I, K}}(undef, length(fars))

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    _foreach(enumerate(fars)) do (idx, far)
        @views blk = zeros(
            K,
            length(rpivots[far[1]].loc),
            length(rpivots[far[2]].loc),
        )

        @views matrixassembler(
            blk,
            rpivots[far[1]].glo[rpivots[far[1]].loc],
            rpivots[far[2]].glo[rpivots[far[2]].loc],
        )
        lowrankblocks[idx] = H2MatrixBlock(
            blk,
            far[1],
            far[2],
        )
    end

    return lowrankblocks
end

##

a = Dict(5=>[1, 2])

for (key, val) in (a)
    print(key)
end