using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using JLD2
using LinearAlgebra
using NestedCrossApproximation


Γ = meshsphere(1.0, 0.02)
op = Helmholtz3D.singlelayer()
space = lagrangec0d1(Γ)
##
A = assemble(op, space, space)
##
jldsave("sphere_HH3D.HS_0.02.jld2"; A)

##


tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = computeinteractions(blktree,  η=1.0)
clusterblocks = Vector{NestedCrossApproximation.PivotBlocks{Int, Float64}}(undef, length(tree.nodes))
@views farblkassembler = BEAST.blockassembler(
    op, space, space
)

@views function farassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    farblkassembler(tdata,sdata,store)
end

interactionlist = NestedCrossApproximation.sort_interactions(
    length(tree.nodes), fars; testortrial=1
)
clusterlink = FastBEAST.cluster_link(tree)


compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting(space.pos),
    NestedCrossApproximation.PCAPivoting(space.pos),
    tol=1e-4,
    maxrank=70
);

am = NestedCrossApproximation.allocate_pca_memory_rm(
    Float64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    true; 
    maxrank=70, 
)

for level in clusterlink
    for nodeidx in level
        childrange = FastBEAST.child_link(tree, nodeidx)

        parent = ClusterTrees.parent(tree, nodeidx)
        if parent != 0 && isassigned(clusterblocks, parent)
            inheritedpivots = clusterblocks[parent].M.σ[clusterblocks[parent].M.M.σ]
        else
            inheritedpivots = Int[]
        end

        tindices, tclustermaps = NestedCrossApproximation.interactionindices(
            tree, nodeidx, interactionlist[nodeidx], inheritedpivots
        )

        if tindices != []
            sindices = value(tree, nodeidx)

            clusterblocks[nodeidx] = NestedCrossApproximation.PivotBlocks(
                NestedCrossApproximation.getcompressedmatrix_rm(
                    farassembler, sindices, tindices, Float64, am[Threads.threadid()], compressor
                ),
                tclustermaps,
                childrange
            )

        end
    end
end
##
err = Float64[]
ind = Int[]
for i in eachindex(clusterblocks)
    if isassigned(clusterblocks, i)
        lrb = clusterblocks[i].M
        lrberr = norm(A[lrb.τ, lrb.σ[lrb.M.σ]]*A[lrb.τ[lrb.M.τ], lrb.σ[lrb.M.σ]]^-1 * 
            A[lrb.τ[lrb.M.τ], lrb.σ]-A[lrb.τ, lrb.σ])/norm(A[lrb.τ, lrb.σ])
        push!(err, lrberr)
        push!(ind, i)
        println(lrberr)
    end
end

argmax(err)
sort(err)
##
lrb = clusterblocks[ind[argmax(err)]].M
norm(A[lrb.τ, lrb.σ]-(A[lrb.τ, lrb.σ[lrb.M.σ]]*A[lrb.τ[lrb.M.τ], lrb.σ[lrb.M.σ]]^-1 * 
A[lrb.τ[lrb.M.τ], lrb.σ]))/norm(A[lrb.τ, lrb.σ])

##
lrb.τ
lrb.σ
##
am = NestedCrossApproximation.allocate_pca_memory_rm(
    Float64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    true; 
    maxrank=70, 
)

lrb2 = NestedCrossApproximation.getcompressedmatrix_rm(
    farassembler, lrb.τ, lrb.σ, Float64, am[Threads.threadid()], compressor
)

comp= FastBEAST.ACAOptions(; tol=1e-4)
am = FastBEAST.allocate_aca_memory(
    Float64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    true; 
    maxrank=50, 
)
lrb2 = NestedCrossApproximation.getcompressedmatrixview(
    farassembler, lrb.τ, lrb.σ, Float64, am[Threads.threadid()], comp
)
lrb2.M.τ
##
norm(A[lrb2.τ, lrb2.σ]-(A[lrb2.τ, lrb2.σ[lrb2.M.σ]]*pinv(A[lrb2.τ[lrb2.M.τ], lrb2.σ[lrb2.M.σ]])* 
A[lrb2.τ[lrb2.M.τ], lrb2.σ]))/norm(A[lrb2.τ, lrb2.σ])
##
perr = [
    1.3908467091004846e-7,
2.444731958697386e-9,
3.0365478208333215e-10,
5.671656853872802e-10,
6.662762841270071e-11,
3.240634721877542e-12,
7.410366858628342e-12,
4.838719158763479e-13]

using Plots
x = Vector(1:length(perr))
f(x) = -6.8277183546534586 -0.6869373851189435*x
plot(x, log10.(perr))
plot!(x, f.(x))

##
points = space.pos[lrb2.σ]
fullp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))
points = space.pos[lrb2.σ[lrb2.M.σ]]
sp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))
points = space.pos[lrb2.τ]
tp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))

using Plots
##

##
plotlyjs()
#scatter(fp[1, :], fp[2, :], fp[3, :])
scatter(fullp[1, :], fullp[2, :], fullp[3, :])
scatter!(sp[1, :], sp[2, :], sp[3, :])
scatter!(tp[1, :], tp[2, :], tp[3, :])