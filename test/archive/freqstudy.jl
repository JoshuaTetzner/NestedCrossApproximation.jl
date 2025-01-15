using BEAST
using FastBEAST
using CompScienceMeshes
using StaticArrays
using LinearAlgebra
using NestedCrossApproximation
using ClusterTrees

λ = 6.0
k = 2 * π / λ

Γ = meshsphere(1.0, 0.05)
op = Maxwell3D.singlelayer(wavenumber=k)
space = raviartthomas(Γ)
tree = create_tree(space.pos, FastBEAST.KMeansTreeOptions(nmin=50))

M = assemble(op, space, space)
##
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree,  η=1.0)
interactionlist = NestedCrossApproximation.sort_interactions(
    length(tree.nodes), fars; testortrial=1
)
function fct(val::F) where F <: Real
    return F(1)/val
end
for leaf in ClusterTrees.leaves(tree)
    τ = value(tree, leaf)
    σ = value(tree, interactionlist[leaf])
    @views function fct(B, x, y)
        B[:,:] = M[τ, σ][x, y]
    end
    lm = LazyMatrix(fct, Vector(1:length(τ)), Vector(1:length(σ)), ComplexF64)
    if size(lm, 1) != 0 && size(lm, 2) != 0
    U, V = aca(lm, tol=1e-4)
    println("ACA: ", norm(U*V-M[τ, σ])/norm(M[τ, σ]))

    am_rm = NestedCrossApproximation.allocate_pca_memory_rm(ComplexF64 ,length(τ), length(σ), maxrank=60)
    
    function fctP(r::F) where F <: Real
        return abs((k^2*r^2 + 2 * im * k * r - 2) / r)
    end
    
    pivstrat = NestedCrossApproximation.PCAPivoting(
        fctP, tree.nodes[leaf].node.data.ct, space.pos[σ]
    )
    
    U, V, r, c = NestedCrossApproximation.pca_rm(
        lm,
        am_rm,
        pivstrat,
        tol=1e-4,
    );
    r = τ[r]
    c = σ[c]
    println("PCA: ", norm(M[τ, c]*M[r, c]^-1*M[r, σ]-M[τ, σ])/norm(M[τ, σ]))
end
end

