using FastBEAST
using BEAST
using ClusterTrees
using CompScienceMeshes
using NestedCrossApproximation
import FastBEAST.NminClusterTrees.NminTree
using ThreadsX
##
using StaticArrays

function updatefilld!(pos::Union{Vector{SVector{3, F}}, SubArray}, h::Vector{F}, pivot::Int) where {F}
    for k in eachindex(h)
        if h[k] > norm(pos[k] - pos[pivot])
            h[k] = norm(pos[k] - pos[pivot])
        end
    end
end

function getfillpivots!(
    pos::Union{Vector{SVector{3, F}}, SubArray}, 
    pivots::Union{Vector{Int}, SubArray},
    ref::SVector{3, F}, 
    maxrank::Int
) where F

    w = 1 ./ (norm(pos .- Scalar(ref)))
    pivots[1] = argmax(w)
    h = norm.(pos .- Scalar(pos[pivots[1]])) 
    for i in 2:maxrank
        @views pivots[i] = argmax(w .* h)
        @views updatefilld!(pos, h, pivots[i])
    end
end

function pivotselection2(
    tree::NminTree{D}, 
    fars::Vector{Vector{Tuple{Int, Int}}}, 
    pos::Vector{SVector{3, F}};
    maxrank=40, multithreading=true
) where {F, D}
    pivots = zeros(Int, length(tree.nodes), maxrank) 
    interactionlist = NestedCrossApproximation.sort_interactions(
       length(tree.nodes), fars; testortrial=1
    )   
    clusterlink = FastBEAST.cluster_link(tree)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    #every cluster with parent interactions needs pivots
    for lcluster in clusterlink
        _foreach(lcluster) do (cluster) 
            cidcs = value(tree, interactionlist[cluster])
            parent = ClusterTrees.parent(tree, cluster)
            if interactionlist[parent] != []
                append!(cidcs, pivots[parent, :])
            end
            if cidcs != []
                refct = tree.nodes[cluster].node.data.ct
                @views getfillpivots!(pos[cidcs], pivots[cluster, 1:maxrank], refct, maxrank)
            end
        end
    end

    return pivots
end

function pivotselection2c(
    tree::NminTree{D}, 
    fars::Vector{Vector{Tuple{Int, Int}}}, 
    pos::Vector{SVector{3, F}};
    maxrank=40, multithreading=true
) where {F, D}
    pivots = zeros(Int, maxrank, length(tree.nodes)) 
    interactionlist = NestedCrossApproximation.sort_interactions(
       length(tree.nodes), fars; testortrial=1
    )   
    clusterlink = FastBEAST.cluster_link(tree)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    #every cluster with parent interactions needs pivots
    for lcluster in clusterlink
        _foreach(lcluster) do (cluster) 
            cidcs = value(tree, interactionlist[cluster])
            parent = ClusterTrees.parent(tree, cluster)
            if interactionlist[parent] != []
                append!(cidcs, pivots[1:maxrank, parent])
            end
            if cidcs != []
                refct = tree.nodes[cluster].node.data.ct
                @views getfillpivots!(pos[cidcs], pivots[1:maxrank, cluster], refct, maxrank)
            end
        end
    end

    return pivots
end

function pivotselection(
    tree::NminTree{D}, 
    fars::Vector{Vector{Tuple{Int, Int}}}, 
    pivstrat::NestedCrossApproximation.PivStrat;
    maxrank=40, multithreading=true
) where D

    pivots = Vector{NestedCrossApproximation.Pivot}(undef, length(tree.nodes)) 
    interactionlist = NestedCrossApproximation.sort_interactions(
       length(tree.nodes), fars; testortrial=1
    )   
    clusterlink = FastBEAST.cluster_link(tree)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    
    #every cluster with parent interactions needs pivots
    for lcluster in clusterlink
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
                pivots[cluster] = NestedCrossApproximation.Pivot(pivot, cidcs[pivot])
            end
        end
    end

    return pivots
end


##
Γ=meshsphere(1.0, 0.02)

op = Helmholtz3D.singlelayer()
space = lagrangecxd0(Γ)
space.pos
tree = FastBEAST.create_tree(space.pos, KMeansTreeOptions(nmin=40))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree,  η=1.0)

##
function fct(x::F) where F
    return F(1)/x
end

piv = NestedCrossApproximation.MyPivoting(fct, space.pos)
@time pivotselection(tree, fars, piv, multithreading=true);
@time pivotselection2(tree, fars, space.pos, multithreading=true);
@time pivotselection2c(tree, fars, space.pos, multithreading=true);

