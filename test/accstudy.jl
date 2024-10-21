using FastBEAST
using BEAST
using NestedCrossApproximation
using StaticArrays
using CompScienceMeshes
using LinearAlgebra
using FLoops
using ClusterTrees

function oneoverR(x, y)
    if x == y
        return 0
    else
        return 1/norm(x-y)
    end
end

Γ = meshsphere(1.0, 0.06)
op = Helmholtz3D.singlelayer()#KernelFunction{Float64}(oneoverR)
space = lagrangec0d1(Γ)#FastBEAST.PointSpace{Float64}(Γ.vertices[2:end])
A = assemble(op, space, space)
##
#=
A = zeros(Float64, length(space.pos), length(space.pos))
@floop for i in eachindex(space.pos)
    for j in eachindex(space.pos)
        A[i, j] = oneoverR(space.pos[i], space.pos[j])
    end
end=#
##
fct(x) = 1/x
for tol in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    println()
    println("Tolerance: ", tol)
    println()
    compressor = NestedCrossApproximation.PCAOptions(
        NestedCrossApproximation.PCAPivoting(fct, space.pos),
        NestedCrossApproximation.PCAPivoting(fct, space.pos),
        tol=tol, 
        maxrank=100
    );
    tree = create_tree(space.pos, KMeansTreeOptions(nmin=100))

    ##
    @time h2matpca = NestedCrossApproximation.GalerkinNCA(
        op, space;
        tree=tree, compressor=compressor, verbose=false,
        momentquadstrat=BEAST.defaultquadstrat(op, space, space));
    println(tol, ", Acc: ", estimate_reldifference(h2matpca, A, tol=1e-3))
    println("Storage: ", storage(h2matpca))

    ##
    @time h2mataca = NestedCrossApproximation.GalerkinNCA(
        op, space,
        compressor=FastBEAST.ACAOptions(tol=tol, maxrank=100), tree=tree, verbose=false,
        momentquadstrat=BEAST.defaultquadstrat(op, space, space));
    println(tol, ", Acc: ", estimate_reldifference(h2mataca, A, tol=1e-3))
    println("Storage: ", storage(h2mataca))
end


##
tree = create_tree(space.pos, KMeansTreeOptions(nmin=100))
@time h2mataca = NestedCrossApproximation.PetrovGalerkinNCA(
    op, space, space,
    compressor=FastBEAST.ACAOptions(tol=1e-12, maxrank=100), 
    testtree=tree, trialtree=tree, verbose=false);
println(1e-6, ", Acc: ", estimate_reldifference(h2mataca, A, tol=1e-3))
##
for i2o in h2mataca.i2otranslator
    println(length(i2o.Z.τ))
end
##
rb = h2mataca.i2otranslator[1].row_basis
cb = h2mataca.i2otranslator[1].col_basis
r = value(tree, rb)
sr = h2mataca.i2otranslator[1].Z.τ
c = value(tree, cb)
sc = h2mataca.i2otranslator[1].Z.σ

blk = A[r, c]

norm(A[r, sc]*pinv(A[sr, sc])*A[sr, c]-blk)/norm(blk)
##
M = fullmat(h2mataca)
value(tree, 10)
##
h2mataca.fars
err = [[] for i in h2mataca.fars]
for lfars in eachindex(h2mataca.fars)
    for far in h2mataca.fars[lfars]
        r = value(tree, far[1])
        c = value(tree, far[2])
        push!(err[lfars], norm(A[r, c]-M[r, c])/norm(A[r, c]))
    end
end

##
example = A[]

##
Γs = meshrectangle(1.0, 1.0, 0.1)
Γu = translate(meshrectangle(4.0, 1.0, 0.1), SVector(4.0, 0.0, 0.0))
Γt = translate(meshrectangle(1.0, 1.0, 0.1), SVector(4.0, 0.0, 0.0))
op = Helmholtz3D.singlelayer()#KernelFunction{Float64}(oneoverR)
spaces = lagrangec0d1(Γs)
spaceu = lagrangec0d1(Γu)
spacet = lagrangec0d1(Γt)

Af = assemble(op, spacet, spaces)
A = assemble(op, spaces, spaceu)

@views function fct(B, x, y)
    B[:,:] = A[x, y]
end
##
tol = 1e-10
lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)
U, V, r, c = aca(lm, tol=tol, svdrecompress=false, maxrank=80)
println("Error ACA: ", norm(U*V - A)/norm(A))

# basis 
TU = U * V[:, c]
Ub = TU * (TU[r, :])^-1 
norm((TU[r, :])^-1*TU[r, :]-I(size(TU, 2)))/norm(I(size(TU, 2)))
##
cols = []
for (ind, p) in enumerate(spaceu.pos)
    if norm(p) < 5.0
        push!(cols, ind)
    end
end
#
@views function fctf(B, x, y)
    B[:,:] = A[:, cols][x, y]
end
lm = LazyMatrix(fctf, Vector(1:size(A[:, cols], 1)), Vector(1:size(A[:, cols], 2)), Float64)
Uf, Vf, rf, cf = aca(lm, tol=tol, svdrecompress=false, maxrank=80)
println("Error ACA: ", norm(Uf*Vf - A[:, cols])/norm(A[:, cols]))

# basis 
TV = Uf[rf, :] * Vf
Vb = (TV[:, cf])^-1 * TV 
norm((TV[:, cf])^-1*TV[:, cf]-I(size(TV, 1)))/norm(I(size(TV, 2)))

norm(Ub*A[r, cols[cf]]*Vb - A[:, cols])/norm(A[:, cols])
