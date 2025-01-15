using BEAST
using FastBEAST
using CompScienceMeshes
using ClusterTrees
using LinearAlgebra
using NestedCrossApproximation


Γ = meshsphere(1.0, 0.04)
λ = 5
k = 2 + pi / λ  
op = Maxwell3D.singlelayer(wavenumber=k)
space = raviartthomas(Γ)
##
A = assemble(op, space, space)
##
#jldsave("sphere_HH3D.HS_0.02.jld2"; A)

##


tree = create_tree(space.pos, KMeansTreeOptions(nmin=50, nchildren=2))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = computeinteractions(blktree,  η=1.0)
clusterblocks = Vector{NestedCrossApproximation.PivotBlocks{Int, ComplexF64}}(undef, length(tree.nodes))
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

interactionlist
compressor = NestedCrossApproximation.PCAOptions(
    NestedCrossApproximation.PCAPivoting2(space.pos),
    NestedCrossApproximation.PCAPivoting2(space.pos),
    tol=1e-8,
    maxrank=70
);

am = NestedCrossApproximation.allocate_pca_memory_rm(
    ComplexF64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    false; 
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
            refcenter = tree.nodes[nodeidx].node.data.ct
            clusterblocks[nodeidx] = NestedCrossApproximation.PivotBlocks(
                NestedCrossApproximation.getcompressedmatrix_rm(
                    farassembler, sindices, tindices, ComplexF64, am[1], compressor, refcenter=refcenter
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
err[argmax(err)] 
sort(err)
##
ind[argmax(err)]
lrb = clusterblocks[ind[argmax(err)]].M
norm(A[lrb.τ, lrb.σ]-(A[lrb.τ, lrb.σ[lrb.M.σ]]*A[lrb.τ[lrb.M.τ], lrb.σ[lrb.M.σ]]^-1 * 
A[lrb.τ[lrb.M.τ], lrb.σ]))/norm(A[lrb.τ, lrb.σ])

##
lrb.τ
lrb.M.σ
##
am = NestedCrossApproximation.allocate_pca_memory_rm(
    ComplexF64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    false; 
    maxrank=70, 
)
sum(space.pos[lrb.τ])./length(lrb.τ)
refcenter = tree.nodes[729].node.data.ct

lrb3 = NestedCrossApproximation.getcompressedmatrix_rm(
    farassembler, lrb.τ, lrb.σ, ComplexF64, am[1], compressor; refcenter=sum(space.pos[lrb.τ])./length(lrb.τ))

comp= FastBEAST.ACAOptions(; tol=1e-8)
am = FastBEAST.allocate_aca_memory(
    ComplexF64, 
    length(value(tree, 1)), 
    length(value(tree, 1)), 
    false; 
    maxrank=50, 
)
lrb2 = NestedCrossApproximation.getcompressedmatrixview(
    farassembler, lrb.τ, lrb.σ, ComplexF64, am[1], comp
)
lrb3.M.σ  
lrb.M.τ 
##
norm(A[lrb2.τ, lrb2.σ]-(A[lrb2.τ, lrb2.σ[lrb2.M.σ]]*pinv(A[lrb2.τ[lrb2.M.τ], lrb2.σ[lrb2.M.σ]])* 
A[lrb2.τ[lrb2.M.τ], lrb2.σ]))/norm(A[lrb2.τ, lrb2.σ])
norm(A[lrb2.τ, lrb2.σ]-(A[lrb3.τ, lrb3.σ[lrb3.M.σ]]*pinv(A[lrb3.τ[lrb3.M.τ], lrb3.σ[lrb3.M.σ]])* 
A[lrb3.τ[lrb3.M.τ], lrb3.σ]))/norm(A[lrb2.τ, lrb2.σ])
##

##
points = space.pos[lrb2.σ]
fullp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))
points = space.pos[lrb.σ[lrb.M.σ]]
sp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))
points = space.pos[lrb2.σ[lrb2.M.σ]]
sp2 = reshape([point[i] for point in points for i in 1:3], (3, length(points)))
points = space.pos[lrb2.τ]
tp = reshape([point[i] for point in points for i in 1:3], (3, length(points)))

using Plots
##

##
plotlyjs()
#scatter(fp[1, :], fp[2, :], fp[3, :])
scatter(fullp[1, :], fullp[2, :], fullp[3, :])
scatter!(sp[1, :], sp[2, :], sp[3, :], label="PCA")
scatter!(sp2[1, :], sp2[2, :], sp2[3, :], label="ACA")
scatter!(tp[1, :], tp[2, :], tp[3, :])